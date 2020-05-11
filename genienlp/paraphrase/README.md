## Get Started

1. Install AWS CLI
2. Download the paraphraing model from s3://almond-research/sinaj/models/schemaorg/paraphrase/gpt2-medium-parabank-2/
```
The command for AWS CLI is aws s3 sync s3://almond-research/sinaj/models/schemaorg/paraphrase/gpt2-medium-parabank-2/ path/to/your/local/model/directory/
```

3. Install `genienlp`
4. To test your installation works, create input.tsv file with sentences you want to paraphrase, one sentence in each line. run the following command.

```
genienlp run-paraphrase --input_file input.tsv --is_cased --model_name_or_path path/to/your/local/model/directory/ --temperature 0 --input_column 0
```
This will run `genienlp/paraprhase/run_generation.py`. You should see paraphrases of your sentences in stdout.

## Command-Line Arguments

Running `genienlp run-paraphrase --help` will give you a description of all the arguments that are available. Here we describe some of the important ones. You can see their implementation in `genienlp/paraphrase/run_generation.py`.

Input files have tab-separated value (`.tsv`) format. You should use `--input_column`, and optionally `--prompt_column`, `--gold_column` and `--thingtalk_column` to specify the meaning of each column. `--input_column` contains the phrases you want to paraphrase, you can include the first part of the paraphrase as a hint in `--prompt_column` column. `--gold_column` should be gold (human) paraphrases if you have access to them, and will be used to calculate BLEU and exact-match scores, if not provided, these will be calculated using `--input_column`. `--thingtalk_column` helps the model fix the capitalization of parameter values in the input phrase by looking at its ThingTalk representation, e.g. "Show me chinese restaurants." becomes "Show me **C**hinese restaurants."

If `--skip_heuristics` is set, inputs will be fed into the model without any changes.

Set `is_cased` if the paraprhasing model you are using is case-sensitive.

These arguments specify the hyperparameters of the decoding algorithm.
`--temperature`, `--top_k`, `--top_p` and `--repetition_penalty` change the token distribution that the decoding algorithm samples from. `--temperature=0` is equivalent to greedy decoding. `--repetition_penalty` reduces the probability of tokens that have already been seen in the full model output (includes model inputs as well).
`--num_beams` if greater than 1, enables beam search. 
`--no_repeat_ngram_siz` if bigger than 0, will prevent repetition of n-grams in the full model output (this includes model inputs as well, so for example setting this to 1 will result in a paraphrase that has no tokens in common with the input).
