#!/bin/bash
set -ex

if [ -n "${S3_MODEL_DIR}" ]; then
  aws s3 sync --no-progress --exclude '*/dataset/*' --exclude '*/cache/*' --exclude 'iteration_*.pth' --exclude '*_optim.pth' "${S3_MODEL_DIR}" ./model
fi
if [ -n "${S3_DATABASE_DIR}" ]; then
  aws s3 sync --no-progress "${S3_DATABASE_DIR}" ./database
fi
exec genienlp $@
