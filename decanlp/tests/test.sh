#!/usr/bin/env bash

set -e
set -x
SRCDIR=`dirname $0`

# functional tests

function delete {
    rm -rf $1
}

mkdir -p $SRCDIR/embeddings

for v in glove.6B.50d charNgram ; do
    for f in vectors itos table ; do
        wget -c "https://parmesan.stanford.edu/glove/${v}.txt.${f}.npy" -O $SRCDIR/embeddings/${v}.txt.${f}.npy
    done
done

    TMPDIR=`pwd`
    workdir=`mktemp -d $TMPDIR/decaNLP-tests-XXXXXX`

    i=0

    for hparams in "" ; do

        # train
        pipenv run decanlp train --train_tasks almond  --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_$i --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $SRCDIR/embeddings --small_glove --no_commit
        # with curriculum
        pipenv run decanlp train --train_tasks almond  --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_"$i"_curriculum --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $SRCDIR/embeddings --small_glove --no_commit --use_curriculum
        # with grammar
        pipenv run decanlp train --train_tasks almond  --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_"$i"_grammar --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $SRCDIR/embeddings --small_glove --no_commit --thingpedia $SRCDIR/dataset/thingpedia-8strict.json --almond_grammar full.bottomup
        # with thingpedia as context
        pipenv run decanlp train --train_tasks almond_with_thingpedia_as_context  --train_iterations 4 --preserve_case --save_every 2 --log_every 2 --val_every 2 --save $workdir/model_"$i"_context_switched --data $SRCDIR/dataset/  $hparams --exist_ok --skip_cache --root "" --embeddings $SRCDIR/embeddings --small_glove --no_commit --thingpedia $SRCDIR/dataset/thingpedia-8strict.json

        # greedy decode
        pipenv run decanlp predict --tasks almond --evaluate test --path $workdir/model_$i --overwrite --eval_dir $workdir/model_$i/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings
        pipenv run decanlp predict --tasks almond --evaluate test --path $workdir/model_"$i"_curriculum  --overwrite --eval_dir $workdir/model_"$i"_curriculum/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings
        pipenv run decanlp predict --tasks almond --evaluate test --path $workdir/model_"$i"_grammar --overwrite --eval_dir $workdir/model_"$i"_grammar/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings --thingpedia $SRCDIR/dataset/thingpedia-8strict.json
        pipenv run decanlp predict --tasks almond --evaluate test --path $workdir/model_"$i"_context_switched  --overwrite --eval_dir $workdir/model_"$i"_context_switched/eval_results/ --data $SRCDIR/dataset/ --embeddings $SRCDIR/embeddings --thingpedia $SRCDIR/dataset/thingpedia-8strict.json

        # export prediction results
        pipenv run python3 $SRCDIR/../utils/post_process_decoded_results.py --original_data $SRCDIR/dataset/almond/test.tsv --gold_program $workdir/model_$i/eval_results/test/almond.gold.txt --predicted_program $workdir/model_$i/eval_results/test/almond.txt --output_file $workdir/model_$i/results.tsv
        pipenv run python3 $SRCDIR/../utils/post_process_decoded_results.py --original_data $SRCDIR/dataset/almond/test.tsv --gold_program $workdir/model_"$i"_curriculum/eval_results/test/almond.gold.txt --predicted_program $workdir/model_"$i"_curriculum/eval_results/test/almond.txt --output_file $workdir/model_"$i"_curriculum/results.tsv
        pipenv run python3 $SRCDIR/../utils/post_process_decoded_results.py --original_data $SRCDIR/dataset/almond/test.tsv --gold_program $workdir/model_"$i"_grammar/eval_results/test/almond.gold.txt --predicted_program $workdir/model_"$i"_grammar/eval_results/test/almond.txt --output_file $workdir/model_"$i"_grammar/results.tsv
        pipenv run python3 $SRCDIR/../utils/post_process_decoded_results.py --original_data $SRCDIR/dataset/almond/test.tsv --gold_program $workdir/model_"$i"_context_switched/eval_results/test/almond.gold.txt --predicted_program $workdir/model_"$i"_context_switched/eval_results/test/almond.txt --output_file $workdir/model_"$i"_context_switched/results.tsv


        # check if result files exist
        if [ ! -f $workdir/model_$i/results.tsv ] && [ ! -f $workdir/model_$i/results_raw.tsv ]; then
            echo "File not found!"
            exit
        fi

        i=$((i+1))
    done

trap "delete $workdir" TERM
