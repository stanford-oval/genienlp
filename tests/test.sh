#!/usr/bin/env bash

set -e
set -x
SRCDIR=`dirname $0`

# functional tests


#mkdir ./embeddings
#wget --no-verbose http://nlp.stanford.edu/data/glove.840B.300d.zip ; unzip glove.840B.300d.zip ; mv glove.840B.300d.zip embeddings/ ; rm glove.42B.300d.zip
#wget --no-verbose http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz ; tar -xzvf jmt_pre-trained_embeddings.tar.gz; mv jmt_pre-trained_embeddings embeddings/; rm jmt_pre-trained_embeddings.tar.gz



    TMPDIR=`pwd`
    workdir=`mktemp -d $TMPDIR/decaNLP-tests-XXXXXX`

    i=0

    for hparams in "" ; do

        # train
        pipenv run python3 $SRCDIR/../train.py --train_tasks almond  --train_iterations 4 --preserve_case --save_every 2--log_every 2 --val_every 2 --save $workdir/model_$i --data dataset/  $hparams --exist_ok --skip_cache --no_glove_and_char --elmo 0

        # greedy decode
        pipenv run python3 $SRCDIR/../predict.py --tasks almond --evaluate test --path ~/$workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data dataset/ --no_glove_and_char --elmo 0

        # export prediction results
        pipenv run python3 $SRCDIR/../utils/post_process_decoded_results.py --original_data dataset/test.tsv --gold_program $workdir/model_$i/eval_results/almond.gold.txt --predicted_program $workdir/model_$i/eval_results/almond.txt --output_file $workdir/model_$i/results.tsv

        i=$((i+1))
    done

trap { rm -rf $workdir } EXIT