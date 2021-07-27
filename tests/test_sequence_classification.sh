#!/usr/bin/env bash

. ./tests/lib.sh

genienlp train --train_tasks ood_task --model TransformerForSequenceClassification --pretrained_model distilbert-base-uncased --save $workdir/model --train_iterations 1 --data $SRCDIR/dataset/ood_task/ --skip_cache --force_fast_tokenizer --train_batch_tokens 200
