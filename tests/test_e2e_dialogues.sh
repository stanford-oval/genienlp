#!/usr/bin/env bash

. ./tests/lib.sh

# test e2e dialogue tasks

hparams=(
        "--pretrained_model Helsinki-NLP/opus-mt-en-de"
        "--pretrained_model sshleifer/bart-tiny-random"
        "--pretrained_model Helsinki-NLP/opus-mt-en-de"
        )
tasks=(
      bitod
      bitod_dst
      risawoz
      )

for i in ${!hparams[*]};
do
  # train
  genienlp train \
    $SHARED_TRAIN_HPARAMS \
    --train_tasks ${tasks[i]} \
    --train_batch_tokens 100 \
    --val_batch_size 300 \
    --train_iterations 4 \
    --min_output_length 2 \
    --save $workdir/model_$i \
    --data $SRCDIR/dataset/bitod \
    ${hparams[i]}

  # greedy prediction
  genienlp predict \
    --tasks ${tasks[i]} \
    --evaluate test \
    --path $workdir/model_$i \
    --overwrite \
    --eval_dir $workdir/model_$i/eval_results/ \
    --data $SRCDIR/dataset/bitod \
    --embeddings $EMBEDDING_DIR \
    --extra_metrics e2e_dialogue_score

  # e2e prediction
  genienlp predict \
    --tasks ${tasks[i]} \
    --evaluate test \
    --path $workdir/model_$i \
    --overwrite \
    --eval_dir $workdir/model_$i/e2e_eval_results/ \
    --data $SRCDIR/dataset/bitod \
    --embeddings $EMBEDDING_DIR \
    --extra_metrics e2e_dialogue_score \
    --e2e_dialogue_evaluation

  # check if result file exists
  if ! [[ -f $workdir/model_$i/eval_results/test/${tasks[i]}.tsv || -f $workdir/model_$i/e2e_eval_results/test/e2e_dialogue_preds.json ]] ; then
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
    diff -u $SRCDIR/expected_results/bitod/e2e_dialogue_preds.json $workdir/model_$i/e2e_eval_results/test/e2e_dialogue_preds.json
  fi

  rm -rf $workdir/model_$i $workdir/model_"$i"_exported

done
