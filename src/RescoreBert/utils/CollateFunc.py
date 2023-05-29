import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.functional import softmax

class NBestSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(self)
        self.dataset = dataset
    def __iter__(self):
        for index in range(len(self.dataset)):
            yield (self.dataset[index], index)
    def __len__(self):
        return len(self.dataset)

class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last = False, batch_by_len = True):
        super().__init__(self)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_by_len = batch_by_len

        print(f'batch_size BatchSampler:{self.batch_size}')
    
    def __iter__(self):
        batch = []
        real_batch = 0
        max_len = 0
        for sample in self.sampler:
            data, index = sample
            if (max_len) == 0:
                max_len = data['max_len']

            if (self.batch_by_len and (data['max_len'] != max_len)): # max len not match the current batch
                yield batch
                batch = []
                real_batch = 0
                max_len = data['max_len']

            if (real_batch > 0 and (real_batch + data['nbest']) > self.batch_size): # IF added into batch, the batch size will get over
                yield batch
                batch = []
                real_batch = 0
                max_len = data['max_len']
            
            batch.append(index)
            real_batch += data['nbest']
        
        if (len(batch) > 0) and not self.drop_last:
            yield batch
    
    def __len__(self):
        real_batch = 0
        num = 0
        max_len = 0

        for sample in self.sampler:
            data, _ = sample

            if (max_len == 0):
                max_len = data['max_len']

            if (self.batch_by_len and (data['max_len'] != max_len)):
                num += 1
                real_batch = 0
                max_len = data['max_len']

            if (real_batch > 0 and (real_batch + data['nbest']) > self.batch_size):
                num += 1
                real_batch = 0
                max_len = data['max_len']

            real_batch += data['nbest']
        
        if (real_batch > 0) and not self.drop_last:
            num += 1
        return num
    

class RescoreBert_BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last = False):
        super().__init__(self)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        print(f'batch_size BatchSampler:{self.batch_size}')
    
    def __iter__(self):
        batch = []
        real_batch = 0
        for sample in self.sampler:
            data, index = sample

            if (real_batch > 0 and (real_batch + data['nbest']) > self.batch_size): # IF added into batch, the batch size will get over
                yield batch
                batch = []
                real_batch = 0
            
            batch.append(index)
            real_batch += data['nbest']
        
        if (len(batch) > 0) and not self.drop_last:
            yield batch
    
    def __len__(self):
        real_batch = 0
        num = 0

        for sample in self.sampler:
            data, _ = sample

            if (real_batch > 0 and (real_batch + data['nbest']) > self.batch_size):
                num += 1
                real_batch = 0

            real_batch += data['nbest']
        
        if (real_batch > 0) and not self.drop_last:
            num += 1
        return num


def crossNBestBatch(batch, hard_label = False):
    names = []
    input_ids = []
    attention_mask = []
    nBest = []
    indexes = []
    NBestTokenTypeId = []
    min_lens = []
    max_lens = []

    am_scores = torch.as_tensor([], dtype = torch.float32)
    ctc_scores = torch.as_tensor([], dtype = torch.float32)
    labels = torch.as_tensor([], dtype = torch.float32)
    wers = torch.as_tensor([], dtype = torch.float32)

    utt_count = 0
    for sample in batch:
        # print(f"nbest:{sample['nbest']}")
        names += [sample['name'] for _ in range(sample['nbest'])]
        input_ids += [torch.as_tensor(s, dtype = torch.int64) for s in sample['input_ids']]
        attention_mask += [torch.as_tensor(s, dtype = torch.int64) for s in sample['attention_mask']]
        
        am_scores = torch.cat([am_scores, sample['am_score']], dim = -1)
        ctc_scores = torch.cat([ctc_scores, sample['ctc_score']], dim = -1)

        sort_index = torch.argsort(sample['wer']) # sort index
        if (hard_label):
            label_score = torch.zeros((sample['nbest']), dtype = torch.float32)
            label_score[sort_index[0]] = 1.0
        # rank_scale = torch.as_tensor([(1 / ((2 * rank) + 1)) for rank in range(sample['nbest'])])
        else:
            label_score = torch.reciprocal(1 + sample['wer'])
            label_score[label_score == 1.00] += 10 #extra bonus for correct answer

            for rank, index in enumerate(sort_index):
                label_score[index] += (1 / ((2 * rank) + 1))
            label_score = softmax(label_score, dim = -1)

        labels = torch.cat([labels, label_score])
        wers = torch.cat([wers, sample['wer']])

        nBest.append(sample['nbest'])
        indexes += [rank for rank in range(sample['nbest'])]

        min_lens.append(sample['min_len'])
        max_lens.append(sample['max_len'])

        if (utt_count % 2 == 0):
            NBestTokenTypeId += [0 for _ in range(sample['nbest'])]
        else:
            NBestTokenTypeId += [1 for _ in range(sample['nbest'])]

    input_ids = pad_sequence(input_ids, batch_first = True)
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    am_scores = am_scores.unsqueeze(-1)
    ctc_scores = ctc_scores.unsqueeze(-1)
    nBest = torch.as_tensor(nBest, dtype = torch.int64)

    NBestTokenTypeId = torch.as_tensor(NBestTokenTypeId, dtype = torch.int64)
    cross_attention_mask = torch.zeros(size = (input_ids.shape[0], input_ids.shape[0]), dtype = torch.bool)
    
    start_index = 0
    for length in nBest:
        cross_attention_mask[start_index:start_index + length, start_index:start_index + length] = True
        start_index += length
    
    # print(f'min_lens: {min_lens}\n max_lens: {max_lens}')
    # print(f'input_ids:{input_ids.shape}')

    return {
        "name": names,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "am_score": am_scores,
        "ctc_score": ctc_scores,
        "label": labels,
        'wers': wers,
        'nBestIndex': nBest,
        'indexes': indexes,
        "NBestTokenTypeId": NBestTokenTypeId,
        'crossAttentionMask': cross_attention_mask
    }

def PBertBatch(batch):
    names = []
    input_ids = []
    attention_mask = []
    nBest = []
    indexes = []

    am_scores = torch.as_tensor([], dtype = torch.float32)
    ctc_scores = torch.as_tensor([], dtype = torch.float32)
    wer = torch.as_tensor([], dtype = torch.float32)

    errors = torch.as_tensor([] , dtype = torch.float32)

    for sample in batch:
        # print(f"nbest:{sample['nbest']}")
        names += [sample['name'] for _ in range(sample['nbest'])]
        input_ids += [torch.as_tensor(s, dtype = torch.int64) for s in sample['input_ids']]
        attention_mask += [torch.as_tensor(s, dtype = torch.int64) for s in sample['attention_mask']]
        
        am_scores = torch.cat([am_scores, sample['am_score']], dim = -1)
        ctc_scores = torch.cat([ctc_scores, sample['ctc_score']], dim = -1)

        sort_index = torch.argsort(sample['wer']) # sort index

        # rank_scale = torch.as_tensor([(1 / ((2 * rank) + 1)) for rank in range(sample['nbest'])])
        label_score = torch.reciprocal(1 + sample['wer'])
        errors = torch.cat([errors, sample['wer']], dim = -1)

        label_score[label_score == 1.00] += 10 #extra bonus for correct answer

        for rank, index in enumerate(sort_index):
            label_score[index] += (1 / ((2 * rank) + 1))
        label_score = softmax(label_score, dim = -1)
        wer = torch.cat([wer, label_score])

        nBest.append(sample['nbest'])
        indexes += [rank for rank in range(sample['nbest'])]

    input_ids = pad_sequence(input_ids, batch_first = True)
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    am_scores = am_scores.unsqueeze(-1)
    ctc_scores = ctc_scores.unsqueeze(-1)
    nBest = torch.as_tensor(nBest, dtype = torch.int64)

    return {
        "name": names,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "am_score": am_scores,
        "ctc_score": ctc_scores,
        "label": wer,
        'wers': errors,
        'nBestIndex': nBest,
        'indexes': indexes,
    }

def PBertBatchWithHardLabel(batch):
    names = []
    input_ids = []
    attention_mask = []
    nBest = []
    indexes = []

    am_scores = torch.as_tensor([], dtype = torch.float32)
    ctc_scores = torch.as_tensor([], dtype = torch.float32)
    wer = torch.as_tensor([], dtype = torch.float32)

    for sample in batch:
        # print(f"nbest:{sample['nbest']}")
        names += [sample['name'] for _ in range(sample['nbest'])]
        input_ids += [torch.as_tensor(s, dtype = torch.int64) for s in sample['input_ids']]
        attention_mask += [torch.as_tensor(s, dtype = torch.int64) for s in sample['attention_mask']]
        
        am_scores = torch.cat([am_scores, sample['am_score']], dim = -1)
        ctc_scores = torch.cat([ctc_scores, sample['ctc_score']], dim = -1)

        sort_index = torch.argsort(sample['wer']) # sort index
        label_score = torch.zeros((sample['nbest']), dtype = torch.float32) # 
        label_score[sort_index[0]] = 1  # Add hard label 1
        wer = torch.cat([wer, label_score])

        nBest.append(sample['nbest'])
        indexes += [rank for rank in range(sample['nbest'])]

    input_ids = pad_sequence(input_ids, batch_first = True)
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    am_scores = am_scores.unsqueeze(-1)
    ctc_scores = ctc_scores.unsqueeze(-1)
    nBest = torch.as_tensor(nBest, dtype = torch.int64)

    return {
        "name": names,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "am_score": am_scores,
        "ctc_score": ctc_scores,
        "label": wer,
        'nBestIndex': nBest,
        'indexes': indexes,
    }

class DistrbutedBatchSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas = None, rank = None, shuffle = True, seed = 42, drop_last = False, batch_size = 64):
        super.__init__(dataset = dataset, num_replicas = num_replicas, rank = rank, shuffle = shuffle, seed = seed, drop_last = drop_last)
        self.batch_size = batch_size
    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = BatchSampler(self.dataset, batch_size=self.batch_size, indices=indices)
        return iter(batch_sampler)
    def __len__(self):
        return self.num_samples

def recogBatch(batch):
    names = []
    indexs = []
    input_ids = []
    attention_mask = []

    for sample in batch:
        indexs.append(sample['index'])
        names.append(sample['name'])
        input_ids.append(torch.tensor(sample['input_ids'], dtype = torch.long))
        # token_type_ids.append(torch.tensor(sample['token_type_ids'], dtype = torch.long))
        attention_mask.append(torch.tensor(sample['attention_mask'], dtype = torch.long))
    
    input_ids = pad_sequence(input_ids, batch_first = True)
    # token_type_ids = pad_sequence(token_type_ids, batch_first = True)
    attention_mask = pad_sequence(attention_mask, batch_first = True)

    return(
        {
            "input_ids": input_ids,
            # "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "name": names,
            "index": indexs
        }
    )

def RescoreBertBatch(batch):
    names = []
    input_ids = []
    attention_masks = []
    scores = []
    labels = []
    errs = []
    wers = []
    avg_errors = []

    names = [sample['name'] for sample in batch]
    for sample in batch:
        names.append(sample['name'])
        input_ids.append(torch.tensor(sample['input_ids'], dtype = torch.int64))
        attention_masks.append(torch.tensor(sample['attention_mask'], dtype = torch.int64))
        scores.append(sample['score'])
        labels.append(sample['mlm_score'])
        errs.append(sample['err'])
        wers.append(sample['wer'])
        avg_errors.append(sample['avg_err'])
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    scores = torch.tensor(scores, dtype = torch.float32)
    labels = torch.tensor(labels, dtype = torch.float32)
    wers = torch.tensor(wers, dtype = torch.float32)
    avg_errors = torch.tensor(avg_errors, dtype = torch.float32)

    return {
        "name": names,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "score": scores,
        "labels": labels,
        "err": errs,
        "wer": wers,
        "avg_error": avg_errors
    }

def MDTrainBatch(batch):

    input_ids = [torch.as_tensor(sample['input_ids'], dtype = torch.int64) for sample in batch]
    attention_mask = [torch.as_tensor(sample['attention_mask'], dtype = torch.int64) for sample in batch]
    labels = [sample['mlm_score'] for sample in batch]

    # for sample in batch:
    #     input_ids.append(torch.as_tensor(sample['input_ids'], dtype = torch.int64))
    #     attention_mask.append(torch.as_tensor(sample['attention_mask'], dtype = torch.int64))
    #     labels.append(sample['mlm_score'])

    input_ids = pad_sequence(input_ids, batch_first = True)
    attention_masks = pad_sequence(attention_mask, batch_first = True)

    # print(f"input_ids: {input_ids.shape}")

    labels = torch.as_tensor(
        labels, 
        dtype = torch.float32, 
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")
    )

    # print(f"labels: {labels.shape}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

def MWERBatch(batch):
    names = []
    input_ids = []
    hyps = []
    attention_masks = []
    scores = []
    labels = []
    errs = []
    wers = []
    avg_errors = []
    nbest = []

    for sample in batch:
        # print(sample)
        names += [sample['name'] for _ in sample['hyps']]
        hyps += [hyp for hyp in sample['hyps']]
        input_ids += [torch.tensor(s) for s in sample['input_ids']]
        attention_masks += [torch.tensor(s) for s in sample['attention_mask']]
        scores += sample['score']
        errs += sample['errs']
        avg_errors += sample['avg_err']
        wers += sample['wer']

        labels += sample['rescore']

        nbest.append(sample['nbest'])
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    wers = torch.tensor(wers)
    scores = torch.tensor(scores)
    avg_errors = torch.tensor(avg_errors)

    return {
        "name": names,
        "hyps": hyps,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "score": scores,
        "labels": labels,
        "err": errs,
        "wer": wers,
        "avg_error": avg_errors,
        "nbest": nbest
    }

def MWERTrainBatch(batch):
    input_ids = [
        torch.as_tensor(sample['input_ids'], 
                        dtype = torch.int64, 
                        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")
                    ) for sample in batch]
    
    attention_mask = [
        torch.as_tensor(sample['attention_mask'], 
                        dtype = torch.int64, 
                        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")
                    ) for sample in batch]
    
    labels = [sample['mlm_score'] for sample in batch]
    wers = [sample['errs'] for sample in batch]
    avg_errors = [sample['avg_err'] for sample in batch]

    input_ids = pad_sequence(input_ids, batch_first = True)
    attention_masks = pad_sequence(attention_mask, batch_first = True)

    labels = torch.as_tensor(
        labels, 
        dtype = torch.float32, 
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")
    )
    wers = torch.as_tensor(
        wers,
        dtype = torch.float32
    )



    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "wers":wers
    }

def MWERValidBatch(batch):
    
    pass

def RescoreBertRecogBatch(batch):
    names = []
    input_ids = []
    attention_masks = []
    labels = []
    wers = []
    indexes = []
    top_hyps = []
    for sample in batch:
        names.append(sample['name'])
        input_ids.append(torch.tensor(sample['input_ids'], dtype = torch.int64))
        attention_masks.append(torch.tensor(sample['attention_mask'], dtype = torch.int64))
        labels.append(sample['score'])
        wers.append(sample['wer'])
        indexes.append(sample['index'])
        top_hyps.append(sample['top_hyp'])
    
    assert(len(names) == len(input_ids)), f"input_ids length {len(input_ids)} != name length {len(names)}"
    assert(len(names) == len(attention_masks)), f"attention_masks length {len(attention_masks)} != name length {len(names)}"
    assert(len(names) == len(labels)), f"labels length {len(labels)} != name length {len(names)}"
    assert(len(names) == len(wers)), f"wers length {len(wers)} != name length {len(names)}"
    assert(len(names) == len(top_hyps)), f"top_hyps length {len(top_hyps)} != name length {len(names)}"

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    labels = torch.tensor(labels, dtype = torch.float32)
    wers = torch.tensor(wers, dtype = torch.float32)

    return {
        "name": names,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "wer": wers,
        "index": indexes,
        "top_hyps": top_hyps
    }


def recogMLMBatch(batch):
    names = []
    input_ids = []
    attention_mask = []
    masked_tokens = []
    nBest_index = []
    seq_index = []
    lengths = []

    for sample in batch:
        names.append(sample['name'])
        input_ids.append(sample['input_ids'])
        attention_mask.append(sample['attention_mask'])

        masked_tokens.append(sample['masked_token'])
        nBest_index.append(sample['index'])
        seq_index.append(sample['seq_index'])

        lengths.append(sample['length'])
    
    data_num = len(names)

    assert(len(input_ids) == data_num), f'data_num: {data_num} != len(input_ids): {len(input_ids)}'
    assert(len(attention_mask) == data_num), f'data_num: {data_num} != len(input_ids): {len(attention_mask)}'
    assert(len(seq_index) == data_num), f'data_num: {data_num} != len(input_ids): {len(seq_index)}'
    assert(len(masked_tokens) == data_num), f'data_num: {data_num} != len(input_ids): {len(masked_tokens)}'
    assert(len(nBest_index) == data_num), f'data_num: {data_num} != len(input_ids): {len(nBest_index)}'

    # assert(len(input_ids) > 0), f'{input_ids}'
    
    input_ids = pad_sequence(input_ids, batch_first = True)
    attention_mask = pad_sequence(attention_mask, batch_first = True)

    return (
        {
            "name": names,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_index": seq_index,
            "masked_token": masked_tokens,
            "nBest_index": nBest_index,
            "length": lengths
        }
    )

def adaptionBatch(sample):
    tokens = [torch.tensor(s) for s in sample]

    tokens = pad_sequence(tokens, batch_first=True)

    # segs = pad_sequence(segs, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return tokens, masks

# pll scoring & recognizing
def pllScoringBatch(sample):
    name = [s[0] for s in sample]

    tokens = []
    for s in sample:
        tokens += s[1]

    texts = []
    for s in sample:
        texts += s[2]

    scores = []
    for s in sample:
        scores += s[3]

    ref = [s[4] for s in sample]

    cer = [s[5] for s in sample]

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    # for i, s in enumerate(segs):
    #     segs[i] = torch.tensor(s)

    tokens = pad_sequence(tokens, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return name[0], tokens, texts, masks, torch.tensor(scores), ref, cer

#  MD distillation
def rescoreBertBatch(sample):

    tokens = []
    texts = []

    for s in sample:
        tokens += s[0]

    texts = []
    for s in sample:
        texts += s[1]

    scores = []
    for s in sample:
        scores += s[2]

    cers = []
    for s in sample:
        cers += s[3]
    plls = []
    for s in sample:
        plls += s[4]

    assert len(plls) == len(tokens), f"illegal pll:{len(plls)} != {len(tokens)}"
    plls = torch.tensor(plls)

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    # for i, s in enumerate(segs):
    #     segs[i] = torch.tensor(s)

    tokens = pad_sequence(tokens, batch_first=True)

    # segs = pad_sequence(segs, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return tokens, texts, masks, torch.tensor(scores), torch.tensor(cers), plls

# RescoreBertRecog
def RescoreBertRecog(sample):
    # using with rescoreDataset
    # s[0] : name
    # s[1] : token
    # s[2] : text for hyp
    # s[3] : score
    # s[4] : ref
    # s[5] : err
    names = []
    tokens = []
    scores = []
    texts = []
    refs = []
    cers = []

    for s in sample:
        names += s[0]
        tokens += s[1]
        texts += s[2]
        scores += s[3]
        refs += s[4]
        cers += s[5]

    for i, t in enumerate(tokens):
        tokens[i] = torch.tensor(t)
    # for i, s in enumerate(segs):
    #     segs[i] = torch.tensor(s)

    tokens = pad_sequence(tokens, batch_first=True)

    masks = torch.zeros(tokens.shape, dtype=torch.long)
    masks = masks.masked_fill(tokens != 0, 1)

    return names, tokens, masks, scores, texts, refs, cers
 
def lmBatch(sample):
    tokens = []
    labels = []
    cers = []
    scores = []

    for s in sample:
        cers += s[2]
        scores += s[3]
        for i, t in enumerate(s[0]):
            tokens.append(torch.tensor(t))
        labels.append(torch.tensor(s[1]))

    tokens = pad_sequence(tokens, batch_first = True)
    labels = pad_sequence(labels, batch_first = True)

    attention_masks = torch.zeros(tokens.shape)
    attention_masks[tokens != 0] = 1
        
    label_mask = torch.zeros(labels.shape)
    label_mask[labels != 0] = 1

    return (
        tokens, 
        attention_masks,
        labels,
        label_mask,
        torch.tensor(cers), 
        torch.tensor(scores)
    )

def lmRecogBatch(sample):
    tokens = []
    labels = []
    texts = []
    cers = []
    scores = []
    ref = []

    for s in sample:
        for i, t in enumerate(s[0]):
            tokens.append(torch.tensor(t))

        tokens = pad_sequence(tokens, batch_first = True)

        attention_masks = torch.zeros(tokens.shape)
        attention_masks[tokens != 0] = 1
        
        cers += s[2]
        scores += s[3]
        texts += s[4]
        ref += s[5]
        
    return (
        tokens, 
        attention_masks, 
        torch.tensor(cers), 
        torch.tensor(scores),
        texts,
        ref
    )

def myCollate(batch):

    names = [sample['name'] for sample in batch]

    bert_ids = [torch.as_tensor(sample['bert_input_ids']) for sample in batch]
    gpt_ids = [torch.as_tensor(sample['gpt_input_ids']) for sample in batch]

    bert_masks = [torch.as_tensor(sample['bert_mask']) for sample in batch]
    gpt_masks = [torch.as_tensor(sample['gpt_mask']) for sample in batch]

    am_scores = [sample['am_score'] for sample in batch]
    ctc_scores = [sample['ctc_score'] for sample in batch]

    wers = [sample['wer'] for sample in batch]
    indexs = [sample['index'] for sample in batch]


    bert_ids = pad_sequence(bert_ids, batch_first= True)
    bert_masks = pad_sequence(bert_masks, batch_first= True)
    
    gpt_ids = pad_sequence(gpt_ids, batch_first= True)
    gpt_masks = pad_sequence(gpt_masks, batch_first= True)

    am_scores = torch.as_tensor(am_scores, dtype = torch.float32)
    ctc_scores = torch.as_tensor(ctc_scores, dtype = torch.float32)
    wers = torch.as_tensor(wers, dtype = torch.float32)
    indexs = torch.as_tensor(indexs, dtype = torch.float32)
    
    return {
        "name": names,
        "bert_ids": bert_ids,
        "gpt_ids": gpt_ids,
        "bert_mask": bert_masks,
        "gpt_mask": gpt_masks,
        "am_score": am_scores,
        "ctc_score": ctc_scores,
        "wer": wers,
        "index": indexs
    }