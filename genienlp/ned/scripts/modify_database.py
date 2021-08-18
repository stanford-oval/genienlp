import argparse
from pathlib import Path
from types import SimpleNamespace

from bootleg.symbols.entity_profile import EntityProfile
from bootleg.utils.entity_profile.fit_to_profile import fit_profiles

parser = argparse.ArgumentParser()

parser.add_argument('--database_dir', type=str)

args = parser.parse_args()


database_path = Path(args.database_dir)
entity_profile_cache = database_path / "wiki_entity_data"

ep = EntityProfile.load_from_cache(
    entity_profile_cache, edit_mode=True, verbose=True, no_kg=False, type_systems_to_load=["wiki"]
)


title = "Some New Entity"
# The numeric value is the score associated with the mention
mentions = [["computer", 10.0], ["sparkle device", 12.0]]
wiki_types = ["computer"]
d = {
    "entity_id": "NQ1",
    "mentions": mentions,
    "title": title,
    "types": {"wiki": wiki_types},
}
if not ep.qid_exists("NQ1"):
    ep.add_entity(d)


ep.save(database_path / "new_wiki_entity_data")


# Base model config to modify
old_config_path = str(database_path / "bootleg_uncased/bootleg_config.yaml")
# Provide save path for the new bootleg config yaml file. This can be anywhere.
new_config_save_path = "new_bootleg_config.yaml"
# Base model pth path to modify
model_path = str(database_path / "bootleg_uncased/bootleg_wiki.pth")
# Provice model path for new bootleg model. This can be anywhere
new_model_path = "new_bootleg_model.pth"
# Path where you saved the adjusted entity profile above
new_entity_path = str(database_path / "new_wiki_entity_data")

args = SimpleNamespace(
    # If you would like to use the same vector we used to intialize our model, download the raw_train_metadata (shown below) and set the path here to the init vec.
    init_vec_file=None,
    train_entity_profile=str(entity_profile_cache),
    new_entity_profile=new_entity_path,
    model_path=model_path,
    model_config=old_config_path,
    save_model_path=new_model_path,
    save_model_config=new_config_save_path,
    # If you renamed any QIDs, pass the renaming mapping here as a path to the saved mapping dictionary.
    oldqid2newqid=None,
    # If you do not want us to adjust title embeddings, set to True.
    no_title_emb=False,
    # Bert model to use to generate title embeddings. Set to cased if using cased model.
    bert_model="bert-base-uncased",
    # If you'd like us to user a different cache_dir when loading the Hugging Face BERT model, add that path below.
    bert_model_cache=None,
    # If you want to use the CPU to generate new titles embeddings, set to True.
    cpu=False,
)

fit_profiles(args)
