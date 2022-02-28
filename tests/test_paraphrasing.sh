#!/usr/bin/env bash

. ./tests/lib.sh

i=0

# test almond_natural_seq2seq and almond_paraphrase tasks
for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random"; do

    # train
    genienlp train --train_tasks almond_natural_seq2seq --train_batch_tokens 100 --val_batch_size 100 --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok  --embeddings $EMBEDDING_DIR --no_commit

    # greedy prediction
    genienlp predict --tasks almond_paraphrase --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $EMBEDDING_DIR

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/almond_paraphrase.tsv || test ! -f $workdir/model_$i/eval_results/test/almond_paraphrase.results.json; then
        echo "File not found!"
        exit 1
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
  genienlp train-paraphrase --sort_by_length --input_column 0 --gold_column 1 --train_data_file $workdir/paraphrasing/train.tsv --eval_data_file $workdir/paraphrasing/dev.tsv --output_dir $workdir/"$model_type" --tensorboard_dir $workdir/tensorboard/ --model_type $model_type --do_train --do_eval --evaluate_during_training --overwrite_output_dir --logging_steps 1000 --save_steps 1000 --max_steps 4 --save_total_limit 1 --gradient_accumulation_steps 2 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --num_train_epochs 1 --model_name_or_path $model --overwrite_cache

  # train a second paraphrasing model (testing num_input_chunks)
  genienlp train-paraphrase --sort_by_length --num_input_chunks 2 --input_column 0 --gold_column 1 --train_data_file $workdir/paraphrasing/train.tsv --eval_data_file $workdir/paraphrasing/dev.tsv --output_dir $workdir/"$model_type"_2/ --tensorboard_dir $workdir/tensorboard/ --model_type $model_type --do_train --do_eval --evaluate_during_training --overwrite_output_dir --logging_steps 1000 --save_steps 1000 --max_steps 4 --save_total_limit 1 --gradient_accumulation_steps 2 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --num_train_epochs 1 --model_name_or_path $model --overwrite_cache


  # use it to paraphrase almond's train set
  genienlp run-paraphrase --model_name_or_path $workdir/"$model_type" --length 15 --temperature 0.4 --repetition_penalty 1.0 --num_samples 4 --input_file $SRCDIR/dataset/almond/train.tsv --input_column 1 --output_file $workdir/generated_"$model_type".tsv --task paraphrase

  # check if result file exists
  if test ! -f $workdir/generated_"$model_type".tsv ; then
      echo "File not found!"
      exit 1
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
  genienlp run-paraphrase --model_name_or_path $model --length 15 --temperature 0 --repetition_penalty 1.0 --num_samples 1 --batch_size 3 --input_file $workdir/masked_paraphrasing/dev.tsv --input_column 0 --gold_column 1 --output_file $workdir/generated_"$model_type".tsv  --skip_heuristics --task paraphrase --infill_text --num_text_spans 1 --src_lang en --tgt_lang en

  # create input file for sts filtering
  paste <(cut -f1-2 $workdir/masked_paraphrasing/dev.tsv) <(cut -f2 $workdir/generated_"$model_type".tsv) <(cut -f3 $workdir/masked_paraphrasing/dev.tsv) > $workdir/sts_input_"$model_type".tsv

  # calculate sts score for paraphrases
  genienlp sts-calculate-scores --input_file $workdir/sts_input_"$model_type".tsv --output_file $workdir/sts_output_score_"$model_type".tsv

  # filter paraphrases based on sts score
  genienlp sts-filter --input_file $workdir/sts_output_score_"$model_type".tsv --output_file $workdir/sts_output_"$model_type".tsv --filtering_metric constant --filtering_threshold 0.98


  if test ! -f $workdir/generated_"$model_type".tsv || test ! -f $workdir/sts_output_"$model_type".tsv ; then
      echo "File not found!"
      exit 1
  fi

done

rm -fr $workdir
rm -rf $SRCDIR/torch-shm-fi
