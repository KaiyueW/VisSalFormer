import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="/tmp/kwang67_cache")
print('-------------bert-base-uncased tokenizer loaded-------------')

def padding_fn(data): # use BERT encoder to convert questions into language embeddings.
    img, q, fix, hm, name = zip(*data)
    input_ids = tokenizer(q, return_tensors="pt", padding=True)
    return torch.stack(img), input_ids, torch.stack(fix), torch.stack(hm), name

def padding_fn_eval(data):  # added for the newly defined dataset
    img, q, imgname, label = zip(*data)
    input_ids = tokenizer(list(q), return_tensors="pt", padding=True)
    return torch.stack(img), input_ids, list(imgname), list(label)
    # stack B imgs into a [B, 3, 224, 224]-sized big tensor; imgname is ["10095.png", "10149.png", ...]
