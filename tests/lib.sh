set -e
set -x
SRCDIR=`dirname $0`


on_error () {
  rm -fr $workdir
  rm -rf $SRCDIR/torch-shm-file-*
}

# Test on CPU. Model outputs will be slightly different on GPU, so the tests that check model outputs will fail.
export CUDA_VISIBLE_DEVICES=""
export SENTENCE_TRANSFORMERS_HOME="$EMBEDDING_DIR"
# parameters that are commonly passed to `genienlp train` test cases
export SHARED_TRAIN_HPARAMS="--no_commit --preserve_case --save_every 2 --log_every 2 --val_every 2"

TMPDIR=`pwd`
workdir=`mktemp -d $TMPDIR/genieNLP-tests-XXXXX`
trap on_error ERR INT TERM
