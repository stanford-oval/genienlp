#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# test calibration
for hparams in \
  "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random" ;
do

  # train
  genienlp train \
    --train_tasks almond \
    --train_batch_tokens 100 \
    --val_batch_size 100 \
    --train_iterations 6 \
    --preserve_case \
    --save_every 2 \
    --log_every 2 \
    --val_every 2 \
    --save $workdir/model_$i \
    --data $SRCDIR/dataset/ \
    --exist_ok \
    --embeddings $EMBEDDING_DIR \
    --no_commit \
    $hparams

  # greedy prediction
  genienlp predict \
    --tasks almond \
    --evaluate test \
    --path $workdir/model_$i \
    --overwrite \
    --eval_dir $workdir/model_$i/eval_results/ \
    --data $SRCDIR/dataset/ \
    --embeddings $EMBEDDING_DIR \
    --save_confidence_features \
    --confidence_feature_path $workdir/model_$i/confidences.pkl \
    --mc_dropout_num 10

  # check if confidence file exists
  if test ! -f $workdir/model_$i/confidences.pkl ; then
    echo "File not found!"
    exit 1
  fi

  # calibrate
  genienlp calibrate \
    --confidence_path $workdir/model_$i/confidences.pkl \
    --save $workdir/model_$i \
    --testing \
    --name_prefix test_calibrator

  # check if calibrator exists
  if test ! -f $workdir/model_$i/test_calibrator.calib ; then
    echo "File not found!"
    exit 1
  fi

  echo "Testing the server mode after calibration"
  # single example in server mode
  echo '{"id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}' | genienlp server --path $workdir/model_$i --stdin
  # batch in server mode
  echo '{"id":"dummy_request_id_1", "instances": [{"example_id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}]}' | genienlp server --path $workdir/model_$i --stdin

  rm -rf $workdir/model_$i

  i=$((i+1))
done
