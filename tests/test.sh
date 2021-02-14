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
fi

export SENTENCE_TRANSFORMERS_HOME="$embedding_dir"

TMPDIR=`pwd`
workdir=`mktemp -d $TMPDIR/genieNLP-tests-XXXXXX`
trap on_error ERR INT TERM


i=0

for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/tiny-mbart" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --preprocess_special_tokens" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --almond_detokenize_sentence" \
      "--model TransformerLSTM --pretrained_model bert-base-cased --trainable_decoder_embeddings=50 --num_beams 4 --num_beam_groups 4 --num_outputs 4 --diversity_penalty 1.0" \
      "--model TransformerLSTM --pretrained_model bert-base-multilingual-cased --trainable_decoder_embeddings=50" \
      "--model TransformerLSTM --pretrained_model bert-base-multilingual-cased --trainable_decoder_embeddings=50 --override_question ." \
      "--model TransformerLSTM --pretrained_model xlm-roberta-base --trainable_decoder_embeddings=50" \
      "--model TransformerLSTM --pretrained_model bert-base-cased --trainable_decoder_embeddings=50 --eval_set_name aux" ;
do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond --train_batch_tokens 50 --val_batch_size 50 --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $embedding_dir --no_commit

    # greedy prediction
    pipenv run python3 -m genienlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir --skip_cache

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/almond.tsv ; then
        echo "File not found!"
        exit
    fi

    # test exporting
    pipenv run python3 -m genienlp export --path $workdir/model_$i --output $workdir/model_$i_exported

    if [ $i == 0 ] ; then
      echo "Testing the server mode"
      echo '{"id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}' | pipenv run python3 -m genienlp server --path $workdir/model_$i --stdin
    fi

    rm -rf $workdir/model_$i $workdir/model_$i_exported

    i=$((i+1))
done

# test calibration
for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random" ;
do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond --train_batch_tokens 50 --val_batch_size 50 --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $embedding_dir --no_commit

    # greedy prediction
    pipenv run python3 -m genienlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir --skip_cache --save_confidence_features --confidence_feature_path $workdir/model_$i/confidences.pkl --mc_dropout_num 10

    # check if confidence file exists
    if test ! -f $workdir/model_$i/confidences.pkl ; then
        echo "File not found!"
        exit
    fi

    # calibrate
    pipenv run python3 -m genienlp calibrate --confidence_path $workdir/model_$i/confidences.pkl --save $workdir/model_$i --testing --name_prefix test_calibrator

    # check if calibrator exists
    if test ! -f $workdir/model_$i/test_calibrator.calib ; then
        echo "File not found!"
        exit
    fi

    echo "Testing the server mode after calibration"
    # single example in server mode
    echo '{"id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}' | pipenv run python3 -m genienlp server --path $workdir/model_$i --stdin
    # batch in server mode
    echo '{"id":"dummy_request_id_1", "instances": [{"example_id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}]}' | pipenv run python3 -m genienlp server --path $workdir/model_$i --stdin

    rm -rf $workdir/model_$i

    i=$((i+1))
done


# test NED
for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method bootleg --database_lookup_method ngrams --almond_domains books --bootleg_model bootleg_wiki_types --add_types_to_text append --bootleg_post_process_types" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method bootleg --database_lookup_method ngrams --almond_domains books --bootleg_model bootleg_wiki_types --add_types_to_text no --bootleg_post_process_types" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method naive --database_lookup_method ngrams --almond_domains books --add_types_to_text insert" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method entity-oracle --database_lookup_method ngrams --almond_domains books --add_types_to_text insert" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method type-oracle --database_lookup_method ngrams --almond_domains books --add_types_to_text insert" \
      "--model TransformerLSTM --pretrained_model bert-base-cased --ned_retrieve_method bootleg --database_lookup_method ngrams --almond_domains books --bootleg_model bootleg_wiki_types --add_types_to_text append --bootleg_post_process_types" \
      "--model TransformerLSTM --pretrained_model bert-base-cased --ned_retrieve_method bootleg --database_lookup_method ngrams --almond_domains books --bootleg_model bootleg_wiki_types --add_types_to_text append --bootleg_post_process_types --override_question ." ;
do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond --train_batch_tokens 50 --val_batch_size 50 --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --database_dir $SRCDIR/database/ --data $SRCDIR/dataset/books_v2/ --bootleg_output_dir $SRCDIR/dataset/books_v2/bootleg/  --exist_ok --skip_cache --embeddings $embedding_dir --no_commit --do_ned --database_type json --ned_features type_id type_prob --ned_features_size 1 1 --ned_features_default_val 0 1.0 --num_workers 0 --min_entity_len 2 --max_entity_len 4 $hparams

    # greedy prediction
    pipenv run python3 -m genienlp predict --tasks almond --evaluate valid --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --database_dir $SRCDIR/database/ --data $SRCDIR/dataset/books_v2/ --embeddings $embedding_dir --skip_cache

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/valid/almond.tsv ; then
        echo "File not found!"
        exit
    fi

    # test server for bootleg
    # due to travis memory limitations, uncomment and run this test locally
    # echo '{"id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}' | pipenv run python3 -m genienlp server --path $workdir/model_$i --stdin
    
    rm -rf $workdir/model_$i
    i=$((i+1))
done

# test almond_multilingual task
for hparams in \
      "--model TransformerLSTM --pretrained_model bert-base-multilingual-cased --trainable_decoder_embeddings=50" \
      "--model TransformerLSTM --pretrained_model bert-base-multilingual-cased --trainable_decoder_embeddings=50 --sentence_batching --use_encoder_loss" \
      "--model TransformerLSTM --pretrained_model bert-base-multilingual-cased --trainable_decoder_embeddings=50 --rnn_zero_state cls --almond_lang_as_question" ; do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond_multilingual --train_languages fa+en --eval_languages fa+en --train_batch_tokens 50 --val_batch_size 50 --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $embedding_dir --no_commit

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

# test natural_seq2seq and paraphrase tasks
for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random"; do

    # train
    pipenv run python3 -m genienlp train --train_tasks natural_seq2seq --train_batch_tokens 50 --val_batch_size 50 --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $embedding_dir --no_commit

    # greedy prediction
    pipenv run python3 -m genienlp predict --tasks paraphrase --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $embedding_dir --skip_cache

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/paraphrase.tsv || test ! -f $workdir/model_$i/eval_results/test/paraphrase.results.json; then
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
  rm -rf $workdir/"$model_type"

done


# masked paraphrasing tests
cp -r $SRCDIR/dataset/paraphrasing/ $workdir/masked_paraphrasing/

for model in "sshleifer/bart-tiny-random" "sshleifer/tiny-mbart" ; do

  if [[ $model == *mbart* ]] ; then
    model_type="mbart"
  elif [[ $model == *bart* ]] ; then
    model_type="bart"
  fi

  # use a pre-trained model
  pipenv run python3 -m genienlp run-paraphrase --model_name_or_path $model --length 15 --temperature 0 --repetition_penalty 1.0 --num_samples 1 --batch_size 3 --input_file $workdir/masked_paraphrasing/dev.tsv --input_column 0 --gold_column 1 --output_file $workdir/generated_"$model_type".tsv  --skip_heuristics --task paraphrase --infill_text --num_text_spans 1 --src_lang en --tgt_lang en

  # create input file for sts filtering
  paste <(cut -f1-2 $workdir/masked_paraphrasing/dev.tsv) <(cut -f2 $workdir/generated_"$model_type".tsv) <(cut -f3 $workdir/masked_paraphrasing/dev.tsv) > $workdir/sts_input_"$model_type".tsv

  # calculate sts score for paraphrases
  pipenv run python3 -m genienlp calculate-paraphrase-sts --input_file $workdir/sts_input_"$model_type".tsv --output_file $workdir/sts_output_score_"$model_type".tsv

  # filter paraphrases based on sts score
  pipenv run python3 -m genienlp filter-paraphrase-sts --input_file $workdir/sts_output_score_"$model_type".tsv --output_file $workdir/sts_output_"$model_type".tsv --filtering_metric constant --filtering_threshold 0.98


  if ! [ -f $workdir/generated_"$model_type".tsv && -f $workdir/sts_output_"$model_type".tsv  ]   ; then
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
  pipenv run python3 -m genienlp run-paraphrase --model_name_or_path $model --length 15 --temperature 0 --repetition_penalty 1.0 --num_samples 1 --batch_size 3 --input_file $workdir/translation/en-de/dev_"$base_model"_aligned.tsv --input_column 0 --gold_column 1 --output_file $workdir/generated_"$base_model"_aligned.tsv  --skip_heuristics --att_pooling mean --task translate --src_lang en --tgt_lang de --replace_qp --force_replace_qp --output_attentions

  # check if result file exists and exact match accuracy is 100%
  cut -f2 $workdir/translation/en-de/dev_"$base_model"_aligned.tsv | diff -u - $workdir/generated_"$base_model"_aligned.tsv
  if test ! -f $workdir/generated_"$base_model"_aligned.tsv   ; then
      echo "File not found!"
      exit
  fi

  rm -rf $workdir/generated_"$base_model"_aligned.tsv

done

# test kfserver
for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random" ;
do

    # train
    pipenv run python3 -m genienlp train --train_tasks almond --train_batch_tokens 50 --val_batch_size 50 --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $embedding_dir --no_commit

    # run kfserver in background
    (pipenv run python3 -m genienlp kfserver --path $workdir/model_$i)&
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
