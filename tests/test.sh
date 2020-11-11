#!/usr/bin/env bash
# functional tests

set -e
set -x
SRCDIR=`dirname $0`

on_error () {
    rm -fr $workdir
    rm -rf $SRCDIR/torch-shm-file-*
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
      "--seq2seq_decoder facebook/bart-large --model Bart" \
      "--encoder_embeddings=small_glove+char --decoder_embeddings=small_glove+char" \
      "--encoder_embeddings=bert-base-multilingual-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \
      "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder MQANEncoder" \
      "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \
      "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=BiLSTM --dimension=768" \
      "--encoder_embeddings=xlm-roberta-base --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \
      "--encoder_embeddings=bert-base-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --eval_set_name aux" ;
do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond  --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $embedding_dir --no_commit

    # greedy prediction
    pipenv run python3 -m genienlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir --skip_cache

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/almond.tsv ; then
        echo "File not found!"
        exit
    fi

    if [ $i == 0 ] ; then
      echo "Testing the server mode"
      echo '{"id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}' | pipenv run python3 -m genienlp server --path $workdir/model_$i --stdin
    fi

    rm -rf $workdir/model_$i

    i=$((i+1))
done

# test almond_multilingual task
for hparams in \
      "--encoder_embeddings=bert-base-multilingual-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768" \
      "--encoder_embeddings=bert-base-multilingual-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768 --sentence_batching --train_batch_tokens 4 --val_batch_size 4 --use_encoder_loss" \
      "--encoder_embeddings=bert-base-multilingual-uncased --decoder_embeddings= --trainable_decoder_embeddings=50 --seq2seq_encoder=Identity --dimension=768 --rnn_zero_state cls --almond_lang_as_question" ;

do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond_multilingual --train_languages fa+en --eval_languages fa+en --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $embedding_dir --no_commit

    # greedy decode
    # combined evaluation
    pipenv run python3 -m genienlp predict --tasks almond_multilingual --pred_languages fa+en --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir --skip_cache
    # separate evaluation
    pipenv run python3 -m genienlp predict --tasks almond_multilingual --separate_eval --pred_languages fa+en --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir --skip_cache

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_en.tsv || test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_fa.tsv || test ! -f $workdir/model_$i/eval_results/test/almond_multilingual_fa+en.tsv; then
        echo "File not found!"
        exit
    fi

    rm -rf $workdir/model_$i
    i=$((i+1))
done


# paraphrasing tests
cp -r $SRCDIR/dataset/paraphrasing/ $workdir/paraphrasing/
for model in  "gpt2" "sshleifer/bart-tiny-random" ; do

  if [[ $model == *gpt2* ]] ; then
    model_type="gpt2"
  elif [[ $model == */bart* ]] ; then
    model_type="bart"
  fi

  # train a paraphrasing model for a few iterations
  pipenv run python3 -m genienlp train-paraphrase --sort_by_length --input_column 0 --gold_column 1 --train_data_file $workdir/paraphrasing/train.tsv --eval_data_file $workdir/paraphrasing/dev.tsv --output_dir $workdir/"$model_type" --tensorboard_dir $workdir/tensorboard/ --model_type $model_type --do_train --do_eval --evaluate_during_training --overwrite_output_dir --logging_steps 1000 --save_steps 1000 --max_steps 4 --save_total_limit 1 --gradient_accumulation_steps 2 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --num_train_epochs 1 --model_name_or_path $model --overwrite_cache

  # train a second paraphrasing model (testing num_input_chunks)
  pipenv run python3 -m genienlp train-paraphrase --sort_by_length --num_input_chunks 2 --input_column 0 --gold_column 1 --train_data_file $workdir/paraphrasing/train.tsv --eval_data_file $workdir/paraphrasing/dev.tsv --output_dir $workdir/"$model_type"_2/ --tensorboard_dir $workdir/tensorboard/ --model_type $model_type --do_train --do_eval --evaluate_during_training --overwrite_output_dir --logging_steps 1000 --save_steps 1000 --max_steps 4 --save_total_limit 1 --gradient_accumulation_steps 2 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --num_train_epochs 1 --model_name_or_path $model --overwrite_cache


  # use it to paraphrase almond's train set
  pipenv run python3 -m genienlp run-paraphrase --model_name_or_path $workdir/"$model_type" --length 15 --temperature 0.4 --repetition_penalty 1.0 --num_samples 4 --input_file $SRCDIR/dataset/almond/train.tsv --input_column 1 --output_file $workdir/generated_"$model_type".tsv --task paraphrase

  # check if result file exists
  if test ! -f $workdir/generated_"$model_type".tsv ; then
      echo "File not found!"
      exit
  fi
  rm -rf $workdir/generated_"$model_type".tsv

done


# masked paraphrasing tests
cp -r $SRCDIR/dataset/paraphrasing/ $workdir/masked_paraphrasing/

for model in "sshleifer/bart-tiny-random" ; do

  if [[ $model == *mbart* ]] ; then
    model_type="mbart"
  elif [[ $model == *bart* ]] ; then
    model_type="bart"
  fi

  # use a pre-trained model
  pipenv run python3 -m genienlp run-paraphrase --model_name_or_path $model --length 15 --temperature 0 --repetition_penalty 1.0 --num_samples 1 --batch_size 3 --input_file $workdir/masked_paraphrasing/dev.tsv --input_column 0 --gold_column 1 --output_file $workdir/generated_"$model_type".tsv  --skip_heuristics --task paraphrase --mask_tokens --mask_token_prob 0.15

  if test ! -f $workdir/generated_"$model_type".tsv   ; then
      echo "File not found!"
      exit
  fi

done

rm -fr $workdir
rm -rf $SRCDIR/torch-shm-fi


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
  pipenv run python3 -m genienlp run-paraphrase --model_name_or_path $model --length 15 --temperature 0 --repetition_penalty 1.0 --num_samples 1 --batch_size 3 --input_file $workdir/translation/en-de/dev_"$base_model"_aligned.tsv --input_column 0 --gold_column 1 --output_file $workdir/generated_"$base_model"_aligned.tsv  --skip_heuristics --att_pooling mean --task translate --tgt_lang de --replace_qp --return_attentions

  # check if result file exists and exact match accuracy is 100%
  cut -f2 $workdir/translation/en-de/dev_"$base_model"_aligned.tsv | diff -u - $workdir/generated_"$base_model"_aligned.tsv
  if test ! -f $workdir/generated_"$base_model"_aligned.tsv   ; then
      echo "File not found!"
      exit
  fi

  rm -rf $workdir/generated_"$base_model"_aligned.tsv

done

rm -fr $workdir
rm -rf $SRCDIR/torch-shm-file-*