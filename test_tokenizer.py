#!/usr/bin/env python3
"""
Token长度分析脚本 - 用于比较FLAN数据和医疗数据的tokenization效果
"""
import json
import os
import sys
from pathlib import Path

def test_tokenizer():
    try:
        from transformers import AutoTokenizer
        print("✅ transformers库已安装")
        
        # 尝试加载tokenizer
        model_path = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct"
        if os.path.exists(model_path):
            print(f"✅ 模型路径存在: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer.pad_token_id = 0
            tokenizer.padding_side = "left"
            print("✅ Tokenizer加载成功")
        else:
            print(f"❌ 模型路径不存在: {model_path}")
            print("使用在线模型...")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True)
            tokenizer.pad_token_id = 0
            tokenizer.padding_side = "left"
            print("✅ 在线Tokenizer加载成功")
            
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return False
    
    # 添加prompter
    sys.path.append('.')
    try:
        from utils.prompter import Prompter
        print("✅ Prompter加载成功")
    except Exception as e:
        print(f"❌ Prompter加载失败: {e}")
        return False
    
    # 测试数据
    print("\n" + "="*50)
    print("📊 数据Token长度分析")
    print("="*50)
    
    # 1. 测试FLAN数据
    try:
        with open('data/dataset1/8/local_training_0.json', 'r') as f:
            flan_data = json.load(f)
        
        flan_sample = flan_data[0]
        prompter_alpaca = Prompter('alpaca_short')
        
        flan_full_prompt = prompter_alpaca.generate_prompt(
            flan_sample["instruction"], 
            flan_sample.get("input", None), 
            flan_sample["output"]
        )
        flan_tokens = tokenizer(flan_full_prompt, return_tensors="pt")
        flan_token_count = len(flan_tokens["input_ids"][0])
        
        print(f"📋 FLAN数据样本:")
        print(f"   - 原始instruction长度: {len(flan_sample['instruction'])} 字符")
        print(f"   - 原始output长度: {len(flan_sample['output'])} 字符")
        print(f"   - 完整prompt长度: {len(flan_full_prompt)} 字符")
        print(f"   - Token数量: {flan_token_count}")
        print(f"   - 示例prompt: {flan_full_prompt[:200]}...")
        
    except Exception as e:
        print(f"❌ FLAN数据测试失败: {e}")
    
    # 2. 测试医疗数据
    try:
        with open('dataset1/dataset1/12/local_training_0.json', 'r') as f:
            medical_data = json.load(f)
        
        medical_sample = medical_data[0]
        prompter_medical = Prompter('medical_expert_short')
        
        # 方案1: 只使用Response
        medical_full_prompt_1 = prompter_medical.generate_prompt(
            medical_sample["Question"], 
            None, 
            medical_sample["Response"]
        )
        medical_tokens_1 = tokenizer(medical_full_prompt_1, return_tensors="pt")
        medical_token_count_1 = len(medical_tokens_1["input_ids"][0])
        
        # 方案2: 使用CoT + Response
        combined_output = medical_sample["Complex_CoT"] + "\n\n" + medical_sample["Response"]
        medical_full_prompt_2 = prompter_medical.generate_prompt(
            medical_sample["Question"], 
            None, 
            combined_output
        )
        medical_tokens_2 = tokenizer(medical_full_prompt_2, return_tensors="pt")
        medical_token_count_2 = len(medical_tokens_2["input_ids"][0])
        
        print(f"\n🏥 医疗数据样本:")
        print(f"   - 原始Question长度: {len(medical_sample['Question'])} 字符")
        print(f"   - 原始Response长度: {len(medical_sample['Response'])} 字符")
        print(f"   - 原始CoT长度: {len(medical_sample['Complex_CoT'])} 字符")
        
        print(f"\n   方案1 (只用Response):")
        print(f"   - 完整prompt长度: {len(medical_full_prompt_1)} 字符")
        print(f"   - Token数量: {medical_token_count_1}")
        
        print(f"\n   方案2 (CoT + Response):")
        print(f"   - 完整prompt长度: {len(medical_full_prompt_2)} 字符")
        print(f"   - Token数量: {medical_token_count_2}")
        
        print(f"\n   示例prompt: {medical_full_prompt_1[:300]}...")
        
    except Exception as e:
        print(f"❌ 医疗数据测试失败: {e}")
    
    # 3. 分析和建议
    print(f"\n" + "="*50)
    print("📈 分析结果")
    print("="*50)
    
    try:
        ratio = medical_token_count_1 / flan_token_count
        print(f"🔍 Token长度比较:")
        print(f"   - FLAN数据: {flan_token_count} tokens")
        print(f"   - 医疗数据(方案1): {medical_token_count_1} tokens")
        print(f"   - 医疗数据(方案2): {medical_token_count_2} tokens")
        print(f"   - 比例: 医疗数据是FLAN数据的 {ratio:.1f} 倍")
        
        cutoff_512 = 512
        cutoff_1024 = 1024
        
        print(f"\n⚠️  Cutoff长度分析:")
        print(f"   - cutoff_len=512: FLAN数据{'✅适合' if flan_token_count <= cutoff_512 else '❌超出'}")
        print(f"   - cutoff_len=512: 医疗数据方案1{'✅适合' if medical_token_count_1 <= cutoff_512 else '❌超出'}")
        print(f"   - cutoff_len=512: 医疗数据方案2{'✅适合' if medical_token_count_2 <= cutoff_512 else '❌超出'}")
        
        print(f"\n   - cutoff_len=1024: 医疗数据方案1{'✅适合' if medical_token_count_1 <= cutoff_1024 else '❌超出'}")
        print(f"   - cutoff_len=1024: 医疗数据方案2{'✅适合' if medical_token_count_2 <= cutoff_1024 else '❌超出'}")
        
        print(f"\n💡 建议:")
        if medical_token_count_1 > cutoff_512:
            print(f"   - 建议将cutoff_len从512增加到至少{medical_token_count_1 + 50}")
        if medical_token_count_1 > cutoff_1024:
            print(f"   - 医疗数据过长，建议截断或分段处理")
        
        estimated_memory_ratio = (medical_token_count_1 / flan_token_count) ** 2
        print(f"   - 预估显存使用增加: {estimated_memory_ratio:.1f}倍")
        print(f"   - 建议batch_size减少到原来的1/{int(estimated_memory_ratio)}")
        
    except:
        print("❌ 分析计算失败")
    
    return True

if __name__ == "__main__":
    print("🚀 开始Token长度分析...")
    success = test_tokenizer()
    if success:
        print("\n✅ 分析完成!")
    else:
        print("\n❌ 分析失败!")