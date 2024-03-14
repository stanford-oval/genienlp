<p align="center">
<img style="vertical-align:middle;margin:-10px" width="150" src="https://avatars.githubusercontent.com/u/13667124" />
</p>
<h1 align="center">
<span>GenieNLP</span>
</h1>

<p align="center">
<a href="https://app.travis-ci.com/github/stanford-oval/genienlp"><img src="https://travis-ci.com/stanford-oval/genienlp.svg?branch=master" alt="Build Status"></a>
<a href="https://pypi.org/project/genienlp/"><img src="https://img.shields.io/pypi/dm/genienlp" alt="PyPI Downloads"></a>
<a href="https://github.com/stanford-oval/genienlp/stargazers"><img src="https://img.shields.io/github/stars/stanford-oval/genienlp?style=social" alt="Github Stars"></a>
</p>


GenieNLP is suitable for all NLP tasks, including text generation (e.g. translation, paraphasing), token classification (e.g. named entity recognition) and sequence classification (e.g. NLI, sentiment analysis).


This library contains the code to run NLP models for the [Genie Toolkit](https://github.com/stanford-oval/genie-toolkit) and the [Genie Virtual Assistant](https://genie.stanford.edu/).
Genie primarily uses this library for semantic parsing, paraphrasing, translation, and dialogue state tracking. Therefore, GenieNLP has a lot of extra features for these tasks.

Works with [🤗 models](https://huggingface.co/models) and [🤗 Datasets](https://huggingface.co/datasets).

## Table of Contents <!-- omit in TOC -->

- [Installation](#installation)
- [Usage](#usage)
  - [Training a semantic parser](#training-a-semantic-parser)
  - [Inference on a semantic parser](#inference-on-a-semantic-parser)
  - [Calibrating a trained model](#calibrating-a-trained-model)
  - [Paraphrasing](#paraphrasing)
  - [Translation](#translation)
- [Citation](#citation)


## Installation

GenieNLP is tested with Python 3.8.
You can install the latest release with pip from PyPI:

```bash
pip install genienlp
```

Or from source:
```bash
git clone https://github.com/stanford-oval/genienlp.git
cd genienlp
pip install -e .  # -e means your changes to the code will automatically take effect without the need to reinstall
```

After installation, `genienlp` command becomes available.

Some GenieNLP commands have additional dependencies for plotting and entity detection. If you are using those commands, you can obtain their dependencies by running the following:

```
pip install matplotlib~=3.0 seaborn~=0.9
python -m spacy download en_core_web_sm
```

## Usage

### Training a semantic parser

The general form is:

```bash
genienlp train --train_tasks almond --train_iterations 50000 --data <datadir> --save <model_dir> <flags>
```

The `<datadir>` should contain a single folder called "almond" (the name of the task). That folder should
contain the files "train.tsv" and "eval.tsv" for train and dev set respectively.


To train a BART or other Seq2Seq model, use:

```bash
genienlp train --train_tasks almond --train_iterations 50000 --data <datadir> --save <model_dir> \
  --model TransformerSeq2Seq --pretrained_model facebook/bart-large --gradient_accumulation_steps 20
```

The default batch sizes are tuned for training on a single V100 GPU. Use `--train_batch_tokens` and `--val_batch_size`
to control the batch sizes. See `genienlp train --help` for the full list of options.

### Inference on a semantic parser

In batch mode:

```bash
genienlp predict --tasks almond --data <datadir> --path <model_dir> --eval_dir <output>
```

The `<datadir>` should contain a single folder called "almond" (the name of the task). That folder should
contain the files "train.tsv" and "eval.tsv" for train and dev set respectively. The result of batch prediction
will be saved in `<output>/almond/valid.tsv`, as a TSV file containing ID and prediction.

In interactive mode:

```bash
genienlp server --path <model_dir>
```

Opens a TCP server that listens to requests, formatted as JSON objects containing `id` (the ID of the request),
`task` (the name of the task), `context`, and `question`. The server writes out JSON objects containing `id` and
`answer`. The server listens to port 8401 by default. Use `--port` to specify a different port or `--stdin` to
use standard input/output instead of TCP.


Generate paraphrases:

```bash
genienlp run-paraphrase --model_name_or_path <model_dir> --temperature 0.3 --repetition_penalty 1.0 --num_samples 4 --batch_size 32 --input_file <input_tsv_file> --input_column 1
```

### Translation

Use the following command for training/ finetuning an NMT model:

```bash
genienlp train --train_tasks almond_translate --data <data_directory> --train_languages <src_lang> --eval_languages <tgt_lang> --no_commit --train_iterations <iterations> --preserve_case --save <save_dir> --exist_ok  --model TransformerSeq2Seq --pretrained_model <hf_model_name>
```

We currently support MarianMT, MBART, MT5, and M2M100 models.<br>
To save a pretrained model in genienlp format without any finetuning, set train_iterations to 0. You can then use this model to do inference.

To produce translations for an eval/ test set run the following command:

```bash
genienlp predict --tasks almond_translate --data <data_directory> --pred_languages <src_lang> --pred_tgt_languages <tgt_lang> --path <path_to_saved_model> --eval_dir <eval_dir>  --val_batch_size 4000 --evaluate <valid/test>  --overwrite --silent
```

If your dataset is a document or contains long examples, pass `--translate_example_split` to break the examples down into individual sentences before translation for better results. <br>
To use [alignment](https://aclanthology.org/2020.emnlp-main.481.pdf), pass `--do_alignment` which ensures the tokens between quotations marks in the sentence are preserved during translation.


See `genienlp --help` and `genienlp <command> --help` for more details about each argument.


## Citation

If you use multiTask training in your work, please cite [*The Natural Language Decathlon: Multitask Learning as Question Answering*](https://arxiv.org/abs/1806.08730).

```bibtex
@article{McCann2018decaNLP,
  title={The Natural Language Decathlon: Multitask Learning as Question Answering},
  author={Bryan McCann and Nitish Shirish Keskar and Caiming Xiong and Richard Socher},
  journal={arXiv preprint arXiv:1806.08730},
  year={2018}
}
```

If you use the paraphrasing model (BART or GPT-2 fine-tuned on a paraphrasing dataset), please cite [AutoQA: From Databases to QA Semantic Parsers with Only Synthetic Training Data](https://arxiv.org/abs/2010.04806)

```bibtex
@inproceedings{xu-etal-2020-autoqa,
    title = "{A}uto{QA}: From Databases to {QA} Semantic Parsers with Only Synthetic Training Data",
    author = "Xu, Silei  and Semnani, Sina  and Campagna, Giovanni  and Lam, Monica",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.31",
    pages = "422--434",
}
```

If you use multilingual models such as MarianMT, MBART or MT5 for Seq2Seq tasks, please cite [Localizing Open-Ontology QA Semantic Parsers in a Day Using Machine Translation](https://aclanthology.org/2020.emnlp-main.481/),
[Contextual Semantic Parsing for Multilingual Task-Oriented Dialogues](https://arxiv.org/abs/2111.02574), and the original paper that introduced the model.

```bibtex
@inproceedings{moradshahi-etal-2020-localizing,
    title = "Localizing Open-Ontology {QA} Semantic Parsers in a Day Using Machine Translation",
    author = "Moradshahi, Mehrad and Campagna, Giovanni and Semnani, Sina and Xu, Silei and Lam, Monica",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = November,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.481",
    pages = "5970--5983",
}
```
```bibtex
@article{moradshahi2021contextual,
  title={Contextual Semantic Parsing for Multilingual Task-Oriented Dialogues},
  author={Moradshahi, Mehrad and Tsai, Victoria and Campagna, Giovanni and Lam, Monica S},
  journal={arXiv preprint arXiv:2111.02574},
  year={2021}
}
```

If you use English models such as BART for Seq2Seq tasks, please cite [A Few-Shot Semantic Parser for Wizard-of-Oz Dialogues with the Precise ThingTalk Representation](https://aclanthology.org/2022.findings-acl.317/), and the original paper that introduced the model.

```bibtex
@inproceedings{campagna-etal-2022-shot,
    title = "A Few-Shot Semantic Parser for {W}izard-of-{O}z Dialogues with the Precise {T}hing{T}alk Representation",
    author = "Campagna, Giovanni  and Semnani, Sina  and Kearns, Ryan  and Koba Sato, Lucas Jun  and Xu, Silei  and Lam, Monica",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.317",
    doi = "10.18653/v1/2022.findings-acl.317",
    pages = "4021--4034",
}
```
