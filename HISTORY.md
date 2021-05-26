0.6.0
=====

* Added support for token classification tasks, and added AmbigQA and CrossNER tasks [#128].
* Added support for beam search in server mode [#141].
* Misc bug fixes [#142, #143].
* Updated dependencies [#133, #134, #136, #138, #140, #144, #145].
* Build system and code style fixes [#146].

0.6.0a4
=======

* Translation code is now migrated to the main genienlp codebase under `almond_translate` task [#98, #105].
* Changed data batching code to account for sequence output lengths too [#130]
* Pipenv is removed [#104].
* Bumped Bootleg version to 1.0.1 [#124].
* Bumped Transformers version to 4.5.1 [#114, #127].
* Misc code upgrades and bug fixes [#115, #102, #123].

0.6.0a3
=======

* Loss dropping is now an optional dependency instead of a required one [#96].
* Fixed running Bootleg models in Kubeflow.
* Fixed combining Bootleg and calibration [#99].
* Misc bug fixes [#97, #102].
* Misc build system and test fixes [#101].

0.6.0a2
=======

* Added support for [Bootleg](https://github.com/HazyResearch/bootleg), a state-of-the art
  named entity recognition system. The output of the NER can be fed as auxiliary information
  to the model in embedding or text form [#83, #93].
* Added support for calibration. Calibration is an additional step applied to the output of
  the model to compute a confidence score that can be interpreted as the probability of producing
  a correct parse. Multiple calibrators can be trained, to separately identify likely incorrect
  parses and out-of-domain inputs [#72, #74, #92, #94].
* Added support for inference in Kubeflow, using the new command `genienlp kfserver`, which
  exposes a compatible HTTP interface [#76, #80, #88, #90].
* Preprocessing of inputs can now use the new fast tokenizers from the huggingface library [#66].
* A number of new hyperparameter options were added, includng diverse beam search, loss dropping,
  and a new learning rate schedule [#66].
* Paraphrasing is now a regular task trained with `genienlp train`, and no longer needs a
  different set of commands [#79].
* Misc bug fixes [#67, #68, #69, #70, #71, #85, #95].

0.6.0a1
=======

* Preprocessing of code inputs have changed, and code tokens are no longer treated specially.
  Instead, they are treated as normal words and preprocessed using BPE. This allows using any
  Huggingface tokenizer without changes. Tasks can still define certain tokens that should be
  treated as special tokens. These are either added as new tokens, or further preprocessed
  into non-ambiguous sequences of words.
* Old models (MQAN and baselines) were removed. The GloVe vectors and other non-contextual
  word embeddings were also removed. Old training options that were ineffective or unused
  were removed.
* The internals of the library have been refactored to simplify development allow using any
  Huggingface Seq2Seq or MLM model. As a result, the name of the models have changed: `Seq2Seq`
  is now `TransformerLSTM` and `Bart` is now `TransformerSeq2Seq`. Command-line flags changed as well.

NOTE: due to the change in model names and commnd-line flags, this release is not backward
compatible with models trained with genienlp <= 0.5.0

0.5.0
=====

* Paraphrasing and training was made much faster, with improved GPU usage and by removing
  redundant tokenization in the hot-paraphrase path [#37, #38, #47].
* The transformers library was updated to 4.0; PyTorch dependency increased to 1.6 [#44, #59, #62].
* New models: BART, mBART, mT5. As part of this work, the model code was refactored to be more consistent
  with Huggingface models and generation code [#46, #62].
* Paraphrasing scripts are now proper subcommands of genienlp [#53].
* It is now possible to fine-tune MBart and Marian models for neural machine translation
  and sentence denoising [#54].
* genienlp server can now operate in batch mode, improving GPU utilization [#58].
* Misc bug and documentation fixes [#39, #40, #41, #43, #48, #55, #58].

0.4.0
=====

* Added the ability to run paraphrasing in FP-16 mixed precision mode.
* The dependency on matplotlib and seaborn (used to produce plots for analysis) is now
  optional [#36].

Please see the development releases below for the full list of features in this release.

0.4.0b1
=======

* Fixed handling of CJK characters and combining characters in BERT and XLM tokenizers [#34].

0.4.0a1
=======

* The paraphrase generation code was extended and can now use BART instead of GPT2. It also now
  has the ability to run as a translation task as well (using the Marian models) [#26, #27, #29, #31].
* Added the ability to override the context and the question used as input to the model [#23].
* MultiGPU training was tested and fixed [#25].
* Completed support for beam search, including the ability to return multiple results for a given input [#30].
* Misc bug fixes [#32].

0.3.0
=====

* New option: sentence batching. Multiple sentences with related properties can be batched
  together in microbatches within a larger minibatch [#14, #11].
* Added option to append context and question in a single model input [#18, #20, #22].
* Updated Transformers dependency to 2.9, and fixed compatibility with newer versions [#18, #24].

0.2.0
=====

* No changes since 0.2.0b2.

Please see the development releases below for the full list of features in this release.

0.2.0b2
=======

* Misc bug fixes related to inference time [#12, #13].

0.2.0b1
=======

* Added multilingual Almond tasks [#10].

0.2.0a2
=======

* Misc bug fixes [#8, #9]

0.2.0a1
=======

New features:
* Add new tasks for Almond: almond_dialogue_nlu, almond_dialogue_nlg, almond_dialogue_policy
* Added a new encoder, "Coattention", which encodes the context and question separately, then
  coattends and applies a BiLSTM layer.
* For Coattention and Identity encoder, it is now possible to specify the context and question
  embeddings separately.
* Embeddings in context, question and answer can now be untied, by suffixing the name with '@'
  followed by an unique identifier (e.g. bert-base-uncased@0 and bert-base-uncased@1).
* Added an option to pretrain the context encoder, using MLM objective.
* Added beam search.
* New embedding option: XLM-R (XLM trained with Roberta).
* New task: paraphrasing with GPT2. This is not fully integrated with the other tasks yet,
  but it will in the future.
* New command "genienlp export" can be used to save a trained model for inference.

Incompatible changes:
* The --save flag is now required when calling train

0.1.1
=====

* Fix publishing on pypi

0.1.0
=====

* First release
