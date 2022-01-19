#!/usr/bin/env bash

. ./tests/lib.sh

# train
genienlp train --train_tasks ood_task --model TransformerForSequenceClassification --pretrained_model distilbert-base-uncased --min_output_length 1 --save $workdir/model --train_iterations 20 --save_every 10 --log_every 10 --val_every 10 --data $SRCDIR/dataset/ood/ --skip_cache --force_fast_tokenizer --train_batch_tokens 200 --num_print 0

# greedy prediction
genienlp predict --tasks ood_task --evaluate valid --pred_set_name eval --path $workdir/model --overwrite --eval_dir $workdir/model/eval_results/ --data $SRCDIR/dataset/ood/ --embeddings $EMBEDDING_DIR --skip_cache --val_batch_size 200

# check if result file exists
if test ! -f $workdir/model/eval_results/valid/ood_task.tsv ; then
    echo "File not found!"
    exit 1
fi

# check if predictions matches expected_results
diff -u $SRCDIR/expected_results/sequence_classification/ood_task.tsv $workdir/model/eval_results/valid/ood_task.tsv

rm -rf $workdir/model
