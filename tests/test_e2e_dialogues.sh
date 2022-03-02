#!/usr/bin/env bash

. ./tests/lib.sh

# test e2e dialogue tasks

hparams=(
        "--pretrained_model sshleifer/bart-tiny-random"
        "--pretrained_model sshleifer/bart-tiny-random"
        )
tasks=(
      bitod
      bitod_dst
      )

for i in ${!hparams[*]};
do
    # train
    genienlp train --train_tasks ${tasks[i]} --train_batch_tokens 100 --val_batch_size 300 --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/bitod --exist_ok  --embeddings $EMBEDDING_DIR --no_commit ${hparams[i]}

    # greedy prediction
    genienlp predict --tasks ${tasks[i]} --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/bitod --embeddings $EMBEDDING_DIR  --extra_metrics e2e_dialogue_score

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/${tasks[i]}.tsv ; then
        echo "File not found!"
        exit 1
    fi

    # check export and server mode
    if [ $i == 0 ] ; then
      echo "Testing export"
      genienlp export --path $workdir/model_$i --output $workdir/model_"$i"_exported

      echo "Testing the server mode"
      echo '{"id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}' | genienlp server --path $workdir/model_$i --stdin
    fi

    if [ $i == 0 ] ; then
      # check if predictions matches expected_results
      diff -u $SRCDIR/expected_results/bitod/bitod.tsv $workdir/model_$i/eval_results/test/bitod.tsv
    fi

    rm -rf $workdir/model_$i $workdir/model_"$i"_exported

done
