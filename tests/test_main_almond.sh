#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# test almond task
for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --preprocess_special_tokens --almond_detokenize_sentence" \
      "--model TransformerLSTM --pretrained_model bert-base-cased --trainable_decoder_embeddings=50 --num_beams 4 --num_beam_groups 4 --num_outputs 4 --diversity_penalty 1.0" \
      "--model TransformerLSTM --pretrained_model bert-base-cased --trainable_decoder_embeddings=50  --override_question ." \
      "--model TransformerLSTM --pretrained_model xlm-roberta-base --trainable_decoder_embeddings=50 --eval_set_name aux" ;
do

    # train
    genienlp train --train_tasks almond --train_batch_tokens 100 --val_batch_size 100 --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --embeddings $EMBEDDING_DIR --no_commit

    # greedy prediction
    genienlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $EMBEDDING_DIR --skip_cache

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/test/almond.tsv ; then
        echo "File not found!"
        exit 1
    fi

    # check TransformerSeq2Seq and TransformerLSTM
    if [ $i == 0 ] || [ $i == 2 ] ; then
      echo "Testing export"
      genienlp export --path $workdir/model_$i --output $workdir/model_$i_exported

      echo "Testing the server mode"
      echo '{"id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}' | genienlp server --path $workdir/model_$i --stdin
    fi

    if [ $i == 2 ] ; then
      # check if predictions matches expected_results
      diff -u $SRCDIR/expected_results/almond/bert_base_cased_beam.tsv $workdir/model_$i/eval_results/test/almond.tsv
    fi

    rm -rf $workdir/model_$i $workdir/model_"$i"_exported

    i=$((i+1))
done
