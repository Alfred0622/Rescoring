from transformers import AutoTokenizer
import json
from jiwer import wer

# tokenizer = AutoTokenizer.from_pretrained('stockmark/bart-base-japanese-news', trust_remote_code=True)

setting = ['noLM', 'withLM']
recog_task = ['eval1', 'eval2', 'eval3']

for s in setting:
    for task in recog_task:
        with open(f"/mnt/disk6/Alfred/Rescoring/data/csj/data/{s}/{task}/data.json") as f:
            data_list = json.load(f)

            top_hyps = []
            refs = []

            split_top_tokens = []
            split_ref_tokens = []

            for i, data in enumerate(data_list):
                # print(data.keys())
                top_hyp = data['hyps'][0].replace('<eos>', "").strip()
                top_hyps.append(top_hyp)
                ref = data['ref'].replace('<eos>', "").strip()
                refs.append(ref)

                hyp_token = top_hyp.split()
                ref_token = ref.split()

                new_hyp = []
                for token in hyp_token:
                    # print(f'token:{token}')
                    if (65281 <= ord(token) <= 65374 ): # if 全形
                        new_hyp.append(chr(ord(token) - 65248))
                    else:
                        new_hyp.append(token)
                new_ref = []
                for token in ref_token:
                    if (65281 <= ord(token) <= 65374 ): # if 全形
                        new_ref.append(chr(ord(token) - 65248))
                    else:
                        new_ref.append(token)
                
                split_top_tokens.append(" ".join(new_hyp))
                split_ref_tokens.append(" ".join(new_ref))
                

                # if (' '.join(new_hyp) != top_hyp):
                #     print(f"Org:{top_hyp}\nNew:{' '.join(new_hyp)}\n")
                # if (' '.join(new_ref) != ref):
                #     print(f"Org:{ref}\nNew:{' '.join(new_ref)}\n")

                wer_before = wer(ref, top_hyp)
                wer_after = wer(" ".join(new_ref), " ".join(new_hyp))

                if (wer_before != wer_after):
                    print("Before:")
                    print(f"Org:{top_hyp}\nNew:{' '.join(new_hyp)}\n")

                    print("After")
                    print(f"Org:{ref}\nNew:{' '.join(new_ref)}\n")

                    print(f"=================================================================\n")


            print(f"{task} {s}:{wer(refs, top_hyps)}")
            print(f"After change encoding")
            print(f"{task} {s}:{wer(split_ref_tokens, split_top_tokens)}\n")
                # print(f"=================================================================")
                
                
