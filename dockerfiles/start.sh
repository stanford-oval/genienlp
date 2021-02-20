#!/bin/bash
set -ex

if [ -n "${S3_MODEL_DIR}" ]; then
  aws s3 sync --no-progress --exclude '*/dataset/*' --exclude '*/cache/*' --exclude 'iteration_*.pth' --exclude '*_optim.pth' "${S3_MODEL_DIR}" ./model
fi
if [ -n "${S3_DATABASE_DIR}" ]; then
  aws s3 sync --no-progress "${S3_DATABASE_DIR}" --exclude '*' --include '*es_material*' --include "*${bootleg_model}/*" --include '*emb_data*' --include '*wiki_entity_data*' ./database
fi
exec genienlp $@

