from transformers import AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def tokenize_pairs(pairs, max_len=20):

    src_batch = []
    tgt_batch = []

    for src, tgt in pairs:

        src_ids = tokenizer.encode(src, max_length=max_len, truncation=True)

        tgt_ids = tokenizer.encode(tgt, max_length=max_len-2, truncation=True)

        # add special tokens manually
        tgt_ids = [tokenizer.cls_token_id] + tgt_ids + [tokenizer.sep_token_id]

        # padding
        src_ids = src_ids + [0] * (max_len - len(src_ids))
        tgt_ids = tgt_ids + [0] * (max_len - len(tgt_ids))

        src_batch.append(src_ids)
        tgt_batch.append(tgt_ids)

    return torch.tensor(src_batch), torch.tensor(tgt_batch)