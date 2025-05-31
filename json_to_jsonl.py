import json

with open('./dataset1/dataset1/all_departments_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)  # 原始是列表格式

with open('./dataset1/dataset1/all_departments_test.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

