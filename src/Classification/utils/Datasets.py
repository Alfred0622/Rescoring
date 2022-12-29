import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class ClassificationDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __getitem__(self, idx):
        return self.data[idx]
        
    def __len__(self):
        return len(self.data)

# def get_dataset(json_data, tokenizer, nbest, for_train = True):
#     data_list = []

#     for data in json_data:
#         hyps = data['hyps'][:nbest]

#         data_errs = []
#         for err in data['err'][:nbest]:
#             data_errs.append(err['err'])
#         label_index = torch.argmin(torch.tensor(data_errs)).item()

        
#         if (for_train):
#             data_list.append(
#                 {
#                     "input_ids": hyps,
#                     "labels": label_index
#                 }
#             )
#         else:
#             data_list.append(
#                 {
#                     "name": data['name'],
#                     "input_ids": hyps,
#                     "labels": label_index
#                 }
#             )
#     return ClassificationDataset(data_list)

def get_dataset(json_data, tokenizer, nbest, for_train = True):
    data_list = []
    not_first = 0
    first = 0
    for data in tqdm(json_data):
        hyps = data['hyps'][:nbest]

        output = tokenizer(hyps)

        if ('token_type_ids' in output.keys()):
            input_ids, token_type_ids, attention_mask = output.values()
        else:
            input_ids, attention_mask = output.values()
        
        input_id = []
        segment = []
        masks = []
        data_errs = []
        for err in data['err'][:nbest]:
            data_errs.append(err['err'])
        label_index = torch.argmin(torch.tensor(data_errs)).item()
        
        if (label_index != 0):
            for i, (ids, seg, mask) in enumerate(zip(input_ids, token_type_ids, attention_mask)):
                if (i == 0):
                    input_id += ids
                    segment += seg
                    masks += mask
                else:
                    input_id += ids[1:]
                    masks += mask[1:]
                    if (i % 2 == 1):
                        concat_seg = [1 for _ in seg[1:]]
                    else: concat_seg = [0 for _ in seg[1:]]

                    segment += concat_seg
        
            if (for_train):
                data_list.append(
                        {
                            "input_ids": input_id,
                            "token_type_ids": segment,
                            "attention_mask": masks,
                            "labels": label_index
                        }
                )
            else:
                data_list.append(
                    {
                        "name": data['name'],
                        "input_ids": input_id,
                        "token_type_ids": segment,
                        "attention_mask": masks,
                        "labels": label_index
                    }
                )
        else:
            best_seq = input_ids[0]
            best_seg = token_type_ids[0]
            best_mask = attention_mask[0]

            for index in range(nbest):
                input_id = []
                segment = []
                masks = []

                for i, (ids, seg, mask) in enumerate(zip(input_ids[1:], token_type_ids[1:], attention_mask[1:])):
                    if (i == index):
                        if (index == 0):
                            input_id += best_seq
                            if (index % 2 == 0):
                                segment += [0 for _ in best_seg]
                            else:
                                segment += [1 for _ in best_seg]
                            masks += best_mask
                        else:
                            input_id += best_seq[1:]
                            if (index % 2 == 0):
                                segment += [0 for _ in best_seg[1:]]
                            else:
                                segment += [1 for _ in best_seg[1:]]
                            masks += best_mask[1:]
    
                    if (i == 0 and index != 0):
                        input_id += ids
                        segment += [0 for _ in seg]
                        masks += mask
                    else:
                        input_id += ids[1:]
                        if (i >= index):
                            k = i + 1
                        else: k = i
                        if (k % 2 == 0):
                            segment += [0 for _ in seg[1:]]
                        else:
                            segment += [1 for _ in seg[1:]]
                    
                        masks += mask[1:]
                    
                    if (index == nbest - 1):
                        input_id += best_seq[1:]
                        if (index % 2 == 0):
                            segment += [0 for _ in best_seg[1:]]
                        else:
                            segment += [1 for _ in best_seg[1:]]
                        masks += best_mask[1:]
    
                if (for_train):
                    data_list.append(
                            {
                                "input_ids": input_id,
                                "token_type_ids": segment,
                                "attention_mask": masks,
                                "labels": index
                            }
                    )
                else:
                    data_list.append(
                        {
                            "name": data['name'],
                            "input_ids": input_id,
                            "token_type_ids": segment,
                            "attention_mask": masks,
                            "labels": index
                        }
                    )
    return ClassificationDataset(data_list)