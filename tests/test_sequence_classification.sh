#!/usr/bin/env bash

. ./tests/lib.sh
genienlp train --train_tasks ood_task --model TransformerForSequenceClassification --pretrained_model distilbert-base-uncased --save $workdir/model --train_iterations 100 --save_every 10 --log_every 2 --val_every 10 --data $SRCDIR/dataset/ood/ --skip_cache --force_fast_tokenizer --train_batch_tokens 200 --num_print 0
