import json
import sys
import os

data_path = sys.argv[1]
split_num = sys.argv[2]

print(f'data_path:{data_path}')
print(f'split:1 ~ {split_num}')

combine_list = []
for num in range(1, int(split_num)+1):
    print(f'train:{num}')
    with open(f"{data_path}/train_{num}/data.json") as f:
        
        data_json = json.load(f)

        for data in data_json:
            combine_list.append(data)

print(f'len:{len(combine_list)}')
    
with open(f"{data_path}/data.json", 'w') as f:
    json.dump(combine_list, f, ensure_ascii=False, indent = 1)
        
    
