import os
import sys

sys.path.append("../")
import torch
from model.RescoreBert import RescoreBertAlsem
from model.NBestCrossBert import nBestCrossBert, pBert
from model.ContrastBERT import marginalBert, contrastBert
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BertModel,
    BertForMaskedLM,
    BertTokenizer,
    BertTokenizerFast,
    AutoTokenizer,
    GPT2Tokenizer,
    BertJapaneseTokenizer,
)
from torch.optim import AdamW
from src_utils.getPretrainName import getBertPretrainName

support_dataset = ["aishell", "aishell2", "tedlium2", "librispeech", "csj"]


class RescoreBert(torch.nn.Module):
    def __init__(self, pretrain_name, device):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).to(device)
        self.linear = torch.nn.Linear(768, 1)

        self.l2_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.NLLLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        wers=None,
        avg_error=None,
        mode=None,
    ):
        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output

        # print(f'output:{output.shape}')

        score = self.linear(output)

        score = score.squeeze(-1)
        if labels is not None:
            labels = labels.to(dtype=torch.float32)

            # score[labels == -10000] = -10000

            loss = self.l2_loss(score, labels)
        else:
            loss = None

        return {"score": score, "loss": loss}

    def parameters(self, recurse=True):
        return list(self.bert.parameters()) + list(self.linear.parameters())


def prepare_GPT2(dataset, device):
    if dataset in ["aishell", "aishell2"]:
        model = AutoModelForCausalLM.from_pretrained("ckiplab/gpt2-base-chinese").to(
            device
        )
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    if dataset in ["tedlium2", "librispeech", "tedlium2_conformer"]:
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if dataset in ["csj"]:
        model = AutoModelForCausalLM.from_pretrained(
            "ClassCat/gpt2-base-japanese-v2"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("ClassCat/gpt2-base-japanese-v2")

    return model, tokenizer


def prepare_MLM(dataset, device):
    if dataset in ["aishell", "aishell2"]:
        model = BertForMaskedLM.from_pretrained("bert-base-chinese").to(device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    if dataset in ["tedlium2", "librispeech", "tedlium2_conformer"]:
        model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if dataset in ["csj"]:
        model = BertForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese").to(
            device
        )
        tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese"
        )

    return model, tokenizer


def prepare_RescoreBert(dataset, device):
    if dataset in ["aishell", "aishell2"]:
        model = RescoreBert("bert-base-chinese", device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif dataset in ["tedlium2", "librispeech"]:
        model = RescoreBert("bert-base-uncased", device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif dataset in ["csj"]:
        model = RescoreBert("cl-tohoku/bert-base-japanese", device)
        tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese"
        )

    return model, tokenizer


def prepare_myModel(dataset, lstm_dim, device):
    model = RescoreBertAlsem(dataset, lstm_dim, device)
    gpt2 = None
    if dataset in ["aishell", "aishell2"]:
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        gpt2 = AutoModelForCausalLM.from_pretrained("ckiplab/gpt2-base-chinese").to(
            device
        )
        gpt_tokenizer = AutoTokenizer.from_pretrained("ckiplab/gpt2-base-chinese")
    elif dataset in ["tedlium2", "librispeech"]:
        bert_tokenizer = BertTokenizer.from_pretrained("bert_base_uncased")
        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif dataset in ["csj"]:
        bert_tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        gpt2 = AutoModelForCausalLM.from_pretrained(
            "ClassCat/gpt2-base-japanese-v2"
        ).to(device)
        gpt_tokenizer = AutoTokenizer.from_pretrained("ClassCat/gpt2-base-japanese-v2")

    assert gpt2 is not None, f"{dataset} is not in support dataset:[{support_dataset}]"

    return model, bert_tokenizer, gpt2, gpt_tokenizer


def prepareNBestCrossBert(
    dataset,
    device,
    lstm_dim=512,
    useNbestCross=False,
    trainAttendWeight=False,
    addRes=False,
    fuseType="lstm",
    lossType="KL",
    taskType="WER",
    concatCLS=False,
    dropout=0.1,
    sepTask=True,
    noCLS=True,
    noSEP=False,
):
    pretrain_name = getBertPretrainName(dataset)

    model = nBestCrossBert(
        dataset,
        device,
        lstm_dim=lstm_dim,
        use_fuseAttention=useNbestCross,
        use_learnAttnWeight=trainAttendWeight,
        addRes=addRes,
        fuseType=fuseType,
        taskType=taskType,
        lossType=lossType,
        concatCLS=concatCLS,
        dropout=dropout,
        sepTask=sepTask,
        noCLS=noCLS,
        noSEP=noSEP,
    )

    tokenizer = BertTokenizer.from_pretrained(pretrain_name)

    return model, tokenizer


def preparePBert(dataset, device, hardLabel=False, loss_type="KL", weightByWER="none"):
    pretrain_name = getBertPretrainName(dataset)

    model = pBert(dataset, device, hardLabel, loss_type, weightByWER=weightByWER)

    tokenizer = BertTokenizer.from_pretrained(pretrain_name)

    return model, tokenizer


def prepareContrastBert(args, train_args, mode="CONTRAST"):
    pretrain_name = getBertPretrainName(args["dataset"])

    if mode == "CONTRAST":
        model = contrastBert(
            args,
        )
    elif mode == "MARGIN":
        model = marginalBert(args, margin=float(train_args["margin"]))

    tokenizer = BertTokenizer.from_pretrained(pretrain_name)

    return model, tokenizer
