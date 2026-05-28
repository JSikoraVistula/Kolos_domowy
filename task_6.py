import json

from huggingface_hub import model_info
from transformers import AutoTokenizer


sentence = "Hello, I'm a single sentence."
checkpoint1 = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
checkpoint2 = "daekeun-ml/koelectra-small-v3-nsmc"

tokenizer1 = AutoTokenizer.from_pretrained(checkpoint1)
tokenizer2 = AutoTokenizer.from_pretrained(checkpoint2)

tok1_encoded = tokenizer1.encode(sentence)
tok2_encoded = tokenizer2.encode(sentence)

tok1_tok1_str = tokenizer1.decode(tok1_encoded)
tok2_tok2_str = tokenizer2.decode(tok2_encoded)
tok1_tok2_str = tokenizer2.decode(tok1_encoded)
tok2_tok1_str = tokenizer1.decode(tok2_encoded)

likes_tok1 = int(model_info(checkpoint1).likes)
likes_tok2 = int(model_info(checkpoint2).likes)

print("tok1_tok1_str:", tok1_tok1_str)
print("tok2_tok2_str:", tok2_tok2_str)
print("tok1_tok2_str:", tok1_tok2_str)
print("tok2_tok1_str:", tok2_tok1_str)
print("likes_tok1:", likes_tok1)
print("likes_tok2:", likes_tok2)

output = {
    "tok1_tok1_str": tok1_tok1_str,
    "tok2_tok2_str": tok2_tok2_str,
    "tok1_tok2_str": tok1_tok2_str,
    "tok2_tok1_str": tok2_tok1_str,
    "likes_tok1": likes_tok1,
    "likes_tok2": likes_tok2,
}

with open("task_6.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
