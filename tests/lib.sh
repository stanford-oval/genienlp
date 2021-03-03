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

TMPDIR=`pwd`
workdir=`mktemp -d $TMPDIR/genieNLP-tests-XXXXXX`
trap on_error ERR INT TERM

