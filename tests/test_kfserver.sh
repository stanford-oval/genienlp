#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# test kfserver
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

  # run kfserver in background
  (genienlp kfserver --path $workdir/model_$i)&
  SERVER_PID=$!
  # wait enough for the server to start
  sleep 15

  # send predict request via http
  request='{"id":"123", "task": "generic", "instances": [{"context": "", "question": "what is the weather"}]}'
  status=`curl -s -o /dev/stderr -w "%{http_code}" http://localhost:8080/v1/models/nlp:predict -d "$request"`
  kill $SERVER_PID
  if [[ "$status" -ne 200 ]]; then
    echo "Unexpected http status: $status"
    exit 1
  fi
  rm -rf $workdir/model_$i
  i=$((i+1))
done

rm -fr $workdir
rm -rf $SRCDIR/torch-shm-file-*
