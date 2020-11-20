src_lang=$1
tgt_lang=$2

wget https://object.pouta.csc.fi/Tatoeba-Challenge/${src_lang}-${tgt_lang}.tar
tar -xvf ${src_lang}-${tgt_lang}.tar

cd data/${src_lang}-${tgt_lang}/

gzip -d train.id.gz
gzip -d train.src.gz
gzip -d train.trg.gz


for file in "train" "dev" "test" ; do
  python3 ../../prepare_opus.py ${file}.id ${file}.id.processed
  paste ${file}.id.processed ${file}.src ${file}.trg > ${file}.tsv
done
