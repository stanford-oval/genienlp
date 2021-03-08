#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# translation tests (with `genienlp train`)
mkdir -p $workdir/translation/almond
cp -r $SRCDIR/dataset/translation/en-de $workdir/translation

for model in "Helsinki-NLP/opus-mt-en-de" "sshleifer/tiny-mbart" ; do

    if [[ $model == Helsinki-NLP* ]] ; then
      base_model="marian"
      expected_result='{"bleu": 90.09463792916938}'
    elif [[ $model == *mbart* ]] ; then
      base_model="mbart"
      expected_result='{"bleu": 0}'
    fi

    mv $workdir/translation/en-de/dev_"$base_model"_aligned.tsv $workdir/translation/almond/train.tsv
    cp $workdir/translation/almond/train.tsv $workdir/translation/almond/eval.tsv

    # train
    pipenv run python3 -m genienlp train  --train_tasks almond_translate --train_languages en --train_tgt_languages de --eval_languages en --eval_tgt_languages de --model TransformerSeq2Seq --pretrained_model $model --train_batch_tokens 50 --val_batch_size 50 --train_iterations 10 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $workdir/translation/ --exist_ok --skip_cache --embeddings $embedding_dir --no_commit

    # greedy prediction
    pipenv run python3 -m genienlp predict --tasks almond_translate --evaluate valid --pred_languages en --pred_tgt_languages de --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $workdir/translation/ --embeddings $embedding_dir --skip_cache

    # check if result file exists and matches expected_result
    echo $expected_result | diff -u - $workdir/model_$i/eval_results/valid/almond_translate.results.json

    rm -rf $workdir/generated_"$base_model"_aligned.tsv

    i=$((i+1))
done

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