set -e
set -x
SRCDIR=`dirname $0`


on_error () {
  rm -fr $workdir
  rm -rf $SRCDIR/torch-shm-file-*
}

# allow faster local testing
if test -d $(dirname ${SRCDIR})/.embeddings; then
  EMBEDDING_DIR="$(dirname ${SRCDIR})/.embeddings"
else
  mkdir -p $SRCDIR/embeddings
  EMBEDDING_DIR="$SRCDIR/embeddings"
fi

export SENTENCE_TRANSFORMERS_HOME="$EMBEDDING_DIR"
# parameters that are commonly passed to `genienlp train` test cases
export SHARED_TRAIN_HPARAMS="--embeddings $EMBEDDING_DIR --exist_ok --no_commit --preserve_case --save_every 2 --log_every 2 --val_every 2"

TMPDIR=`pwd`
workdir=`mktemp -d $TMPDIR/genieNLP-tests-XXXXX`
trap on_error ERR INT TERM
