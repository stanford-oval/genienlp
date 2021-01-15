#!/bin/bash
set -ex
if [ -n "${S3_MODEL_DIR}" ]; then
  aws s3 sync --exclude '*/dataset/*' --exclude '*/cache/*' --exclude 'iteration_*.pth' --exclude '*_optim.pth' "${S3_MODEL_DIR}" ./model
  genienlp kfserver --path ./model $*
else
  genienlp $*
fi
