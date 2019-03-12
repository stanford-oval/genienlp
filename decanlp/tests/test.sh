#!/usr/bin/env bash

set -e
set -x
SRCDIR=`dirname $0`

# functional tests

function delete {
    rm -rf $1
}

mkdir -p $SRCDIR/embeddings
curl -O "https://parmesan.stanford.edu/glove/glove.6B.50d.txt.pt" ; mv glove.6B.50d.txt.pt $SRCDIR/embeddings/
curl -O "https://parmesan.stanford.edu/glove/charNgram.txt.pt" ; mv charNgram.txt.pt $SRCDIR/embeddings/

    TMPDIR=`pwd`
    workdir=`mktemp -d $TMPDIR/decaNLP-tests-XXXXXX`

    i=0

    for hparams in "" ; do

        # train
        pipenv run decanlp train --train_tasks almond  --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $SRCDIR/embeddings --small_glove --no_commit
        # with curriculum
        pipenv run decanlp train --train_tasks almond  --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_"$i"_curriculum --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $SRCDIR/embeddings --small_glove --no_commit --use_curriculum


        # greedy decode
        pipenv run decanlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings
        pipenv run decanlp predict --tasks almond --evaluate test --path $workdir/model_"$i"_curriculum  --overwrite --eval_dir $workdir/model_"$i"_curriculum/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings


        # export prediction results
        pipenv run python3 $SRCDIR/../utils/post_process_decoded_results.py --original_data $SRCDIR/dataset/almond/test.tsv --gold_program $workdir/model_$i/eval_results/test/almond.gold.txt --predicted_program $workdir/model_$i/eval_results/test/almond.txt --output_file $workdir/model_$i/results.tsv

        # check if result files exist
        if [ ! -f $workdir/model_$i/results.tsv ] && [ ! -f $workdir/model_$i/results_raw.tsv ]; then
            echo "File not found!"
            exit
        fi

        i=$((i+1))
    done

trap "delete $workdir" EXIT