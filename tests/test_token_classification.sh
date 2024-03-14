#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# test cross_ner task
for hparams in \
  "--crossner_domains music" \
  "--crossner_domains news type_id type_prob --database_dir ${SRCDIR}/database/ --add_entities_to_text append" ;
do

  # train
  genienlp train \
    $SHARED_TRAIN_HPARAMS \
    --train_tasks cross_ner \
    --model TransformerForTokenClassification \
    --pretrained_model bert-base-cased \
    --force_fast_tokenizer \
    --train_batch_tokens 200 \
    --val_batch_size 200 \
    --train_iterations 4 \
    --save $workdir/model_$i \
    --data $SRCDIR/dataset/cross_ner/ \
    $hparams

  # greedy prediction
  genienlp predict \
    --tasks cross_ner \
    --evaluate valid \
    --pred_set_name dev \
    --path $workdir/model_$i \
    --overwrite \
    --eval_dir $workdir/model_$i/eval_results/ \
    --data $SRCDIR/dataset/cross_ner/ \
    --embeddings $EMBEDDING_DIR  \
    --val_batch_size 2000

  # check if result file exists
  if test ! -f $workdir/model_$i/eval_results/valid/cross_ner.tsv ; then
    echo "File not found!"
    exit 1
  fi

  # check if predictions matches expected_results
  diff -u $SRCDIR/expected_results/token_classification/cross_ner_news_$i.tsv $workdir/model_$i/eval_results/valid/cross_ner.tsv

  rm -rf $workdir/model_$i

  i=$((i+1))
done

# test conll2003 task
for hparams in \
  "" ;
do

  # train
  genienlp train \
    $SHARED_TRAIN_HPARAMS \
    --train_tasks conll2003 \
    --crossner_domains music \
    --model TransformerForTokenClassification \
    --pretrained_model bert-base-cased \
    --force_fast_tokenizer \
    --subsample 5 \
    --train_batch_tokens 100 \
    --val_batch_size 100 \
    --train_iterations 4 \
    --save $workdir/model_$i \
    --data $SRCDIR/dataset/cross_ner/ \
    $hparams

  # greedy prediction
  genienlp predict \
    --tasks conll2003 \
    --evaluate valid \
    --pred_set_name validation \
    --subsample 5 --path $workdir/model_$i \
    --overwrite \
    --eval_dir $workdir/model_$i/eval_results/ \
    --data $SRCDIR/dataset/cross_ner/ \
    --embeddings $EMBEDDING_DIR \
    --val_batch_size 2000

  # check if result file exists
  if test ! -f $workdir/model_$i/eval_results/valid/conll2003.tsv ; then
    echo "File not found!"
    exit 1
  fi

  # check if predictions matches expected_results
  diff -u $SRCDIR/expected_results/token_classification/conll2003_$i.tsv $workdir/model_$i/eval_results/valid/conll2003.tsv

  rm -rf $workdir/model_$i

  i=$((i+1))
done
