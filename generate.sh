echo "Start only Global"
python Generate.py \
      --prompt_template 'alpaca_short' \
      --output_file 'out/result.jsonl' \
      --test_file './data/dataset1/flan_test_200_selected_nstrict_1.jsonl'
echo "Global generation  has finished! "
echo "Start Global and local"
python Generate.py \
      --prompt_template 'alpaca_short' \
      --output_file 'out/result2.jsonl' \
      --test_file './data/dataset1/flan_test_200_selected_nstrict_1.jsonl' \
      --local True \
      --local_model_path './lora-7b/8/19/local_output_0/pytorch_model.bin'  \
      --input_file './data/dataset1/8/local_training_0.json' \
      --auto True \
      --max_weight 0.5
echo "local generation has finished!"



