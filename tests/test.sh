#!/usr/bin/env bash

set -e
set -x
SRCDIR=`dirname $0`

# functional tests

function on_error {
    rm -fr $workdir
}

mkdir -p $SRCDIR/embeddings

for v in glove.6B.50d charNgram ; do
    for f in vectors itos table ; do
        wget -c "https://parmesan.stanford.edu/glove/${v}.txt.${f}.npy" -O $SRCDIR/embeddings/${v}.txt.${f}.npy
    done
done

TMPDIR=`pwd`
workdir=$TMPDIR/workdir #`mktemp -d $TMPDIR/genieNLP-tests-XXXXXX`
mkdir -p $workdir/cache
trap on_error ERR INT TERM

i=0
for hparams in \
            "--dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=bert-base-uncased --decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 --rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.1"
            # "--encoder_embeddings=bert-base-multilingual-cased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \
            # "--encoder_embeddings=small_glove+char --decoder_embeddings=small_glove+char" \
            #    "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50" \
            #    "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \
            #    "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=BiLSTM --dimension=768" \
do

    # train
    decanlp train --train_tasks almond  --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --cache $workdir/cache  --root "" --embeddings $SRCDIR/embeddings --no_commit

    # greedy decode
    # decanlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings

    # check if result files exist
    # if test ! -f $workdir/model_$i/eval_results/test/almond.tsv ; then
        # echo "File not found!"
        # exit
    # fi

    i=$((i+1))
done

rm -fr $workdir