#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# test kfserver
for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random" ;
do

    # train
    genienlp train --train_tasks almond --train_batch_tokens 100 --val_batch_size 100 --train_iterations 2 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $EMBEDDING_DIR --no_commit

    # generate a long sequence
    long_sequence=''
    set +x
    for j in {1..2000};
    do
        long_sequence="${long_sequence} XXX"
    done
    set -x

    # test cuda errors
    input="{\"id\": \"test\", \"context\": \"${long_sequence}\", \"question\": \"translate to thingtalk\", \"answer\": \"YYY\"}"
    set +e
    echo ${input} | genienlp server --path $workdir/model_$i --stdin
    exit_code=$?
    set -e

    if [ $exit_code != 100 ] ; then
        echo "Cuda error not caught!"
        exit 1
    fi

    rm -rf $workdir/model_$i
    i=$((i+1))
done

rm -fr $workdir
rm -rf $SRCDIR/torch-shm-file-*
