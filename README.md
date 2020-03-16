# Genie NLP library

[![Build Status](https://travis-ci.com/stanford-oval/genienlp.svg?branch=master)](https://travis-ci.com/stanford-oval/genienlp) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/stanford-oval/genienlp.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/stanford-oval/genienlp/context:python)

This library contains the NLP models for the [Genie](https://github.com/stanford-oval/genie-toolkit) toolkit for
virtual assistants. It is derived from the [decaNLP](https://github.com/salesforce/decaNLP) library by Salesforce,
but has diverged significantly.

The library is suitable for all NLP tasks that can be framed as Contextual Question Answering, that is, with 3 inputs:
- text or structured input as _context_
- text input as _question_
- text or structured output as _answer_

As the [decaNLP paper](https://arxiv.org/abs/1806.08730) shows, many different NLP tasks can be framed in this way.
Genie primarily uses the library for semantic parsing, dialogue state tracking, and natural language generation 
given a formal dialogue state, and this is what the models work best for.

## Installation

genienlp is available on PyPi. You can install it with:
```bash
pip3 install genienlp
```

After installation, a `genienlp` command becomes available.

Likely, you will also want to download the word embeddings ahead of time:

```bash
genienlp cache-embeddings --embeddings glove+char -d <embeddingdir>
```

## Usage

Train a model:
```bash
genienlp train --tasks almond --train_iterations 50000 --embeddings <embeddingdir> --data <datadir> --save <modeldir>
```

Generate predictions:
```bash
genienlp predict --tasks almond --data <datadir> --path <modeldir>
```

Train a paraphrasing model:
```bash
genienlp train-paraphrase --train_data_file <train_data_file> --eval_data_file <dev_data_file> --output_dir <modeldir> --model_type gpt2 --do_train --do_eval --evaluate_during_training --logging_steps 1000 --save_steps 1000 --max_steps 40000 --save_total_limit 2 --gradient_accumulation_steps 16 --per_gpu_eval_batch_size 4 --per_gpu_train_batch_size 4 --num_train_epochs 1 --model_name_or_path <gpt2/gpt2-medium/gpt2-large/gpt2-xlarge>
```

Generate paraphrases:
```bash
genienlp run-paraphrase --model_type gpt2 --model_name_or_path <modeldir> --temperature 0.3 --repetition_penalty 1.0 --num_samples 4 --length 15 --batch_size 32 --input_file <input tsv file> --input_column 1
```

See `genienlp --help` and `genienlp <command> --help` for details about each argument.

## Citation

If you use the MultiTask Question Answering model in your work, please cite [*The Natural Language Decathlon: Multitask Learning as Question Answering*](https://arxiv.org/abs/1806.08730).

```bibtex
@article{McCann2018decaNLP,
  title={The Natural Language Decathlon: Multitask Learning as Question Answering},
  author={Bryan McCann and Nitish Shirish Keskar and Caiming Xiong and Richard Socher},
  journal={arXiv preprint arXiv:1806.08730},
  year={2018}
}
```

If you use the BERT-LSTM model (Identity encoder + MQAN decoder), please cite [_Schema2QA: Answering Complex Queries on the Structured Web with a Neural Model_](https://arxiv.org/abs/2001.05609)

```bibtex
@article{Xu2020Schema2QA,
  title={Schema2QA: Answering Complex Queries on the Structured Web with a Neural Model},
  author={Silei Xu and Giovanni Campagna and Jian Li and Monica S. Lam},
  journal={arXiv preprint arXiv:2001.05609},
  year={2020}
}
```
