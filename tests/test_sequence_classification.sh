#!/usr/bin/env bash

. ./tests/lib.sh

# Test ood task
# train
genienlp train --train_tasks ood_task --model TransformerForSequenceClassification --pretrained_model distilbert-base-uncased --min_output_length 1 --save $workdir/model --train_iterations 20 --save_every 10 --log_every 10 --val_every 10 --data $SRCDIR/dataset/ood/  --force_fast_tokenizer --train_batch_tokens 200 --num_print 0

# greedy prediction
genienlp predict --tasks ood_task --evaluate valid --pred_set_name eval --path $workdir/model --overwrite --eval_dir $workdir/model/eval_results/ --data $SRCDIR/dataset/ood/ --embeddings $EMBEDDING_DIR  --val_batch_size 200

# check if result file exists
if test ! -f $workdir/model/eval_results/valid/ood_task.tsv ; then
    echo "File not found!"
    exit 1
fi

# check if predictions matches expected_results
diff -u $SRCDIR/expected_results/sequence_classification/ood_task.tsv $workdir/model/eval_results/valid/ood_task.tsv

rm -rf $workdir/model


# Test bitod_error_cls task
# train
genienlp train --train_tasks bitod_error_cls --model TransformerForSequenceClassification --pretrained_model distilbert-base-uncased --min_output_length 1 --save $workdir/model_error/ --train_iterations 100 --save_every 50 --log_every 50 --val_every 50 --data $SRCDIR/dataset/bitod_error/  --force_fast_tokenizer --train_batch_tokens 200 --num_print 0

# greedy prediction
genienlp predict --tasks bitod_error_cls --evaluate valid --pred_set_name valid --path $workdir/model_error --overwrite --eval_dir $workdir/model_error/eval_results/ --data $SRCDIR/dataset/bitod_error/ --embeddings $EMBEDDING_DIR  --val_batch_size 200

# check if result file exists
if test ! -f $workdir/model_error/eval_results/valid/bitod_error_cls.tsv ; then
    echo "File not found!"
    exit 1
fi

# check if predictions matches expected_results
diff -u $SRCDIR/expected_results/sequence_classification/bitod_error_cls.tsv $workdir/model_error/eval_results/valid/bitod_error_cls.tsv

rm -rf $workdir/model_error
