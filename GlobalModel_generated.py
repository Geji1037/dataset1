import os

import fire
import gradio as gr
import torch
import transformers
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

import sys
import tty
import termios

def wait_for_key():
    """Wait for a key press on the console and return it."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig,  LlamaTokenizer,AutoTokenizer,AutoModelForCausalLM
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

class EvalDataset(Dataset):
    def __init__(self, file, prompter, tokenizer, max_len=512):
        self.prompter = prompter
        self.tokenizer = tokenizer
        with open(file, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = line.strip()
        ques = json.loads(line)
        sample = ques['instruction']
        prompt = self.prompter.generate_prompt(sample, None)
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"][0]
        return prompt, sample

def writeFile(s, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'a+',encoding='utf-8') as f1:
        f1.write(s+'\n')

def main(
    load_8bit: bool = False,
    base_model: str = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct",
    lora_weights_path: str = "/root/.cache/modelscope/hub/models/LLM-Research/Fedray/FedDPA/code/lora-7b/8/19/adapter_model.bin",
    lora_config_path: str= "/root/.cache/modelscope/hub/models/LLM-Research/Fedray/FedDPA/code/lora-7b/8/", # provide only the file path, excluding the file name 'adapter_config.json'
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "127.0.0.1",
    share_gradio: bool = False,
    output_file: str="",
    test_file: str="",
    batched: bool = True,
    local: bool = False,
    local_model_path: str = "",
    weight: float = 0.5,
    input_file: str="",
    auto: bool = False,
    half: bool = True,
    max_weight: float = -1,
    instance_num: int = 5,
    emb_type: str="last",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code = True)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"
    gpu_count = torch.cuda.device_count()

    if not lora_weights_path.endswith(".bin"):
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code = True,
                quantization_config = None,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code = True,
                quantization_config = None,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, 
                device_map={"": device}, 
                trust_remote_code = True,
                quantization_config = None,
                low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code = True,
            quantization_config = None,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig.from_pretrained(lora_config_path)
        if gpu_count<3:
            print(gpu_count)
            lora_weights = torch.load(lora_weights_path,map_location=lambda storage, loc: storage.cuda(0))
        else:
            lora_weights = torch.load(lora_weights_path)
        model = PeftModel(model, config)
        
        if local:
            # local_adapters_weights = torch.load(local_model_path)
            # print('local', local_adapters_weights['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'][0])
            # print('global', lora_weights['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'][0])
            # for k in lora_weights.keys():
            #     if 'lora_A' in k:
            #         lora_weights[k] = (1-weight)*local_adapters_weights[k] + weight*lora_weights[k]
            #     if 'lora_B' in k:
            #         lora_weights[k] = local_adapters_weights[k] + lora_weights[k]
            # # print('combined', lora_weights['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'][0])
            # del local_adapters_weights
            set_peft_model_state_dict(model,lora_weights,"default")
            local_weights = torch.load(local_model_path)

            local_config = LoraConfig.from_pretrained(lora_config_path)
            model.add_adapter("local", local_config)

            set_peft_model_state_dict(model, local_weights,"local")
            model.set_adapter(current_adapter)
            res = evaluate(None,input_ids=input_ids)
            #all_text.extend(text)
            all_res.extend(res)
            #print(all_text)
            #print(all_res)
            #break

    #lines = open('../alpaca-lora-y/data/flan_test_50_selected.jsonl').readlines()
    if auto:
        lists = json.load(open(input_file))
    lines = open(test_file).readlines()
    count=0
    for i,line in enumerate(lines):
        line = line.strip()
        ques = json.loads(line)

        if auto:
            tmpw = 0.5 if half else 1
            if max_weight >0 :
                tmpw = max_weight
            weight = get_weight(ques['instruction'], lists, num=instance_num, w=tmpw, emb_type=emb_type)
            if i==0:
                print("******************", weight,"****************************")
            model.set_local(['local'], [weight,1-weight])
        if not batched:
            res = evaluate(ques['instruction'])
        else:
            res = all_res[i]
        if auto:
            model.unset_local()
        tmp = {}
        tmp['text'] = ques['instruction']
        tmp['answer'] = res
        tmp['category'] = ques['category']
        writeFile(json.dumps(tmp, ensure_ascii=False), save)
        count = count+1
        print('num:', count)
        print("Instruction:", tmp['text'])
        print("Response:", tmp['answer'])
        print("*****************************************************")
        # break



if __name__ == "__main__":
    fire.Fire(main)
