#!/usr/bin/env bash

. ./tests/lib.sh

i=0
# test NED
for hparams in \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method bootleg --database_lookup_method ngrams --almond_domains thingpedia --bootleg_model bootleg_uncased_mini --add_types_to_text append --bootleg_post_process_types --ned_dump_entity_type_pairs" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method bootleg --database_lookup_method ngrams --almond_domains thingpedia --bootleg_model bootleg_uncased_mini --add_types_to_text no --bootleg_post_process_types" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method naive --database_lookup_method ngrams --almond_domains thingpedia --add_types_to_text insert" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method entity-oracle --database_lookup_method ngrams --almond_domains thingpedia --add_types_to_text insert --ned_dump_entity_type_pairs" \
      "--model TransformerSeq2Seq --pretrained_model sshleifer/bart-tiny-random --ned_retrieve_method type-oracle --database_lookup_method ngrams --almond_domains thingpedia --add_types_to_text insert" \
      "--model TransformerLSTM --pretrained_model bert-base-cased --ned_retrieve_method bootleg --database_lookup_method ngrams --almond_domains thingpedia --bootleg_model bootleg_uncased_mini --add_types_to_text no --bootleg_post_process_types" \
      "--model TransformerLSTM --pretrained_model bert-base-cased --ned_retrieve_method bootleg --database_lookup_method ngrams --almond_domains thingpedia --bootleg_model bootleg_uncased_mini --add_types_to_text append --bootleg_post_process_types --override_context ." ;
do

    # train
    genienlp train --train_tasks almond_dialogue_nlu --train_batch_tokens 100 --val_batch_size 100 --train_iterations 6 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --database_dir $SRCDIR/database/ --data $SRCDIR/dataset/thingpedia_99/ --bootleg_output_dir $SRCDIR/dataset/thingpedia_99/bootleg/  --exist_ok --skip_cache --embeddings $EMBEDDING_DIR --no_commit --do_ned --database_type json --ned_features type_id type_prob --ned_features_size 1 1 --ned_features_default_val 0 1.0 --num_workers 0 --min_entity_len 2 --max_entity_len 4 $hparams

    # greedy prediction
    genienlp predict --tasks almond_dialogue_nlu --evaluate valid --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --database_dir $SRCDIR/database/ --data $SRCDIR/dataset/thingpedia_99/ --embeddings $EMBEDDING_DIR --skip_cache

    # check if result file exists
    if test ! -f $workdir/model_$i/eval_results/valid/almond_dialogue_nlu.tsv ; then
        echo "File not found!"
        exit 1
    fi

    # test server for bootleg
    # due to travis memory limitations, uncomment and run this test locally
    # echo '{"task": "almond_dialogue_nlu", "id": "dummy_example_1", "context": "show me .", "question": "translate to thingtalk", "answer": "now => () => notify"}' | genienlp server --database_dir $SRCDIR/../database/  --path $workdir/model_$i --stdin

    if [ $i == 0 ] ; then
      # check if predictions matches expected_results
      diff -u $SRCDIR/expected_results/NED/bart_tiny_random_0.json $workdir/model_$i/eval_results/valid/almond_dialogue_nlu.results.json
    fi

    rm -rf $workdir/model_$i
    i=$((i+1))
done
