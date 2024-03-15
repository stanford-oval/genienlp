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

### Inference on a semantic parser

In batch mode:

```bash
genienlp predict --tasks almond --data <datadir> --path <model_dir> --eval_dir <output>
```

The `<datadir>` should contain a single folder called "almond" (the name of the task). That folder should
contain the files "train.tsv" and "eval.tsv" for train and dev set respectively. The result of batch prediction
will be saved in `<output>/almond/valid.tsv`, as a TSV file containing ID and prediction.


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
