#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# translation tests
mkdir -p $workdir/translation
cp -r $SRCDIR/dataset/translation/en-de $workdir/translation

for model in "t5-small" "Helsinki-NLP/opus-mt-en-de" ; do

  if [[ $model == *t5* ]] ; then
    base_model="t5"
  elif [[ $model == Helsinki-NLP* ]] ; then
    base_model="marian"
  fi

  # use a pre-trained model
  pipenv run python3 -m genienlp run-paraphrase --model_name_or_path $model --length 15 --temperature 0 --repetition_penalty 1.0 --num_samples 1 --batch_size 3 --input_file $workdir/translation/en-de/dev_"$base_model"_aligned.tsv --input_column 0 --gold_column 1 --output_file $workdir/generated_"$base_model"_aligned.tsv  --skip_heuristics --att_pooling mean --task translate --src_lang en --tgt_lang de --replace_qp --force_replace_qp --output_attentions

  # check if result file exists and exact match accuracy is 100%
  cut -f2 $workdir/translation/en-de/dev_"$base_model"_aligned.tsv | diff -u - $workdir/generated_"$base_model"_aligned.tsv
  if test ! -f $workdir/generated_"$base_model"_aligned.tsv   ; then
      echo "File not found!"
      exit 1
  fi

  rm -rf $workdir/generated_"$base_model"_aligned.tsv

done