#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# test almond_multilingual task
for hparams in \
      "--model TransformerLSTM --pretrained_model bert-base-multilingual-cased --trainable_decoder_embeddings=50" \
      "--model TransformerLSTM --pretrained_model bert-base-multilingual-cased --trainable_decoder_embeddings=50 --sentence_batching --use_encoder_loss" \
      "--model TransformerLSTM --pretrained_model bert-base-multilingual-cased --trainable_decoder_embeddings=50 --rnn_zero_state cls --almond_lang_as_question" ; do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond_multilingual --train_languages fa+en --eval_languages fa+en --train_batch_tokens 50 --val_batch_size 50 --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $EMBEDDING_DIR --no_commit

    # greedy decode
    # combined evaluation
    pipenv run python3 -m genienlp predict --tasks almond_multilingual --pred_languages fa+en --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $EMBEDDING_DIR --skip_cache
    # separate evaluation
    pipenv run python3 -m genienlp predict --tasks almond_multilingual --separate_eval --pred_languages fa+en --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $EMBEDDING_DIR --skip_cache

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_en.tsv || test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_fa.tsv || test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_fa+en.tsv; then
        echo "File not found!"
        exit 1
    fi

    rm -rf $workdir/model_$i
    i=$((i+1))
done
