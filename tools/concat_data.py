# concat the token sequence for comparison
import json
from tqdm import tqdm

setting = ['noLM', 'withLM']

for s in setting:
    with open(f'./data/aishell/train/token/token_{s}_50best.json') as f:
        token_file = json.load(f)
        total_data = len(token_file)
        file_num = 64
        produce_data_num = total_data * 50 * 49

        data_num_per_file = int(produce_data_num / 64)
        
        concat_dict = list()
        for i, data in enumerate(tqdm(token_file)):
            file_num = 1
            temp_dict = data
            temp_list = list()
            label = list()

            for i, first_seq in enumerate(data['token']):
                for j, second_seq in enumerate(data['token']):
                    if (i == j):
                        continue
                    temp_list.append(first_seq + second_seq[1:])
                    if (i > j):
                        label.append(0)
                    else:
                        label.append(1)
            temp_dict['token'] = temp_list
            temp_dict['label'] = label
            concat_dict.append(temp_dict)
            if (i % data_num_per_file == 0):
                if (not os.path.exists(f'./data/aishell/train/token/concat')) :
                    os.makedirs(f'./data/aishell/train/token/concat')
                
                with open(f'./data/aishell/train/token/concat/token_{s}_concat_{file_num}.json', 'w') as fw:
                    json.dump(
                        concat_dict, fw, ensure_ascii = False, indent = 4
                    )
                
                file_num += 1
                concat_dict = dict()
                
    with open(f'./data/aishell/train/token/concat/token_{s}_concat_{file_num}.json', 'w') as fw:
        json.dump(
            concat_dict, fw, ensure_ascii = False, indent = 4
        )