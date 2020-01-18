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
workdir=`mktemp -d $TMPDIR/decaNLP-tests-XXXXXX`
trap on_error ERR INT TERM

i=0
for hparams in "--encoder_embeddings=small_glove+char --decoder_embeddings=small_glove+char" \
               "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50" \
               "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" ; do

    # train
    pipenv run python3 -m decanlp train --train_tasks almond  --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $SRCDIR/embeddings --no_commit

    # greedy decode
    pipenv run python3 -m decanlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings

    # check if result files exist
    if test ! -f $workdir/model_$i/eval_results/test/almond.tsv ; then
        echo "File not found!"
        exit
    fi

    i=$((i+1))
done
