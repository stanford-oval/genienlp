import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

pretrained_model = "dbmdz/electra-large-discriminator-finetuned-conll03-english"
id2label = {
    0: "B-LOC",
    1: "B-MISC",
    2: "B-ORG",
    3: "I-LOC",
    4: "I-MISC",
    5: "I-ORG",
    6: "I-PER",
    7: "O"
}
not_entity_label = "O"

sentence = "I am sarah from londonovich london"
config = AutoConfig.from_pretrained(pretrained_model, cache_dir='.embeddings', num_labels=len(id2label), finetuning_task='ned')
model = AutoModelForTokenClassification.from_pretrained(pretrained_model, cache_dir='.embeddings', config=config)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir='.embeddings', use_fast=True)
tokenized = tokenizer.batch_encode_plus([sentence])
output = model(torch.tensor(tokenized['input_ids']))
logits = output.logits
predictions = torch.argmax(logits, dim=2).tolist()
pred = predictions[0]
labels = [id2label[p] for p in pred]

wordid2label = {}
word_ids = tokenized.encodings[0].word_ids
tokens = tokenized.encodings[0].tokens

for wid, label in zip(word_ids, labels):
    if label != not_entity_label:
        if wid not in wordid2label:
            wordid2label[wid] = label
        
all_entites = []
orig_tokens = sentence.split(" ")
tokens = []
for i, token in enumerate(orig_tokens):
    if i in wordid2label:
        if len(tokens) == 0:
            tokens = [token]
        # join with previous token
        elif i-1 in wordid2label and wordid2label[i-1] == wordid2label[i]:
            tokens.append(token)
        else:
            all_entites.append(" ".join(tokens))
            tokens = [token]
if len(tokens):
    all_entites.append(" ".join(tokens))

from collections import defaultdict
length2entities = defaultdict(list)
for entity in all_entites:
    entity_len = len(entity.split())
    length2entities[entity_len].append(entity)

pass
