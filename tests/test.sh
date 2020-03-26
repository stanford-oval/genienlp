#!/usr/bin/env bash

set -e
set -x
SRCDIR=`dirname $0`

# functional tests

on_error () {
    rm -fr $workdir
}

# allow faster local testing
if test -d $(dirname ${SRCDIR})/.embeddings; then
  embedding_dir="$(dirname ${SRCDIR})/.embeddings"
else
  mkdir -p $SRCDIR/embeddings
  embedding_dir="$SRCDIR/embeddings"

  for v in glove.6B.50d charNgram ; do
      for f in vectors itos table ; do
          wget -c "https://parmesan.stanford.edu/glove/${v}.txt.${f}.npy" -O $SRCDIR/embeddings/${v}.txt.${f}.npy
      done
  done
fi

TMPDIR=`pwd`
workdir=`mktemp -d $TMPDIR/genieNLP-tests-XXXXXX`
trap on_error ERR INT TERM

i=0
for hparams in \
      "--encoder_embeddings=small_glove+char --decoder_embeddings=small_glove+char" \
      "--encoder_embeddings=bert-base-multilingual-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \
      "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50" \
      "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \
      "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=BiLSTM --dimension=768" \
      "--encoder_embeddings=xlm-roberta-base --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" ;
do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond  --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $embedding_dir --no_commit

    # greedy decode
    pipenv run python3 -m genienlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/almond.tsv ; then
        echo "File not found!"
        exit
    fi

    i=$((i+1))
done


# test almond_multilingual task
for hparams in \
      "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \

do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond_multilingual --train_languages fa+en --eval_languages fa+en --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $embedding_dir --no_commit

    # greedy decode
    # combined evaluation
    pipenv run python3 -m genienlp predict --tasks almond_multilingual --pred_languages fa+en --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir
    # separate evaluation
    pipenv run python3 -m genienlp predict --tasks almond_multilingual --separate_eval --pred_languages fa+en --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_en.tsv || test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_fa.tsv || test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_fa+en.tsv; then
        echo "File not found!"
        exit
    fi

    i=$((i+1))
done



# Train a paraphrasing model for a few iterations
cp -r $SRCDIR/dataset/paraphrasing/ $workdir/paraphrasing/
pipenv run python3 -m genienlp train-paraphrase --train_data_file $workdir/paraphrasing/train.txt --eval_data_file $workdir/paraphrasing/dev.txt --output_dir $workdir/gpt2-small-1 --tensorboard_dir $workdir/tensorboard/ --model_type gpt2 --do_train --do_eval --evaluate_during_training --overwrite_output_dir --logging_steps 1000 --save_steps 1000 --max_steps 4 --save_total_limit 1 --gradient_accumulation_steps 1 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --num_train_epochs 1 --model_name_or_path gpt2
# Use it to paraphrase almond's train set
pipenv run python3 -m genienlp run-paraphrase --model_type gpt2 --model_name_or_path $workdir/gpt2-small-1 --length 15 --temperature 0.4 --repetition_penalty 1.0 --num_samples 4 --input_file $SRCDIR/dataset/almond/train.tsv --input_column 1 --output_file $workdir/generated.tsv

# check if result file exists
if test ! -f $workdir/generated.tsv ; then
    echo "File not found!"
    exit
fi

rm -fr $workdir