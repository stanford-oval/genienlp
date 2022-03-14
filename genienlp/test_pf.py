import argparse
import logging
from pprint import pformat

import torch
from parallelformers import parallelize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from genienlp.arguments import check_and_update_generation_args
from genienlp.predict import check_args, set_default_values
from genienlp.tasks.registry import get_tasks
from genienlp.util import get_devices, load_config_json, log_model_size, set_seed

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--path', type=str, required=True, help='Folder to load the model from')
    parser.add_argument(
        '--evaluate',
        type=str,
        required=True,
        choices=['train', 'valid', 'test'],
        help='Which dataset to do predictions for (train, dev or test)',
    )
    parser.add_argument(
        '--pred_set_name',
        default='eval',
        type=str,
        help='Name of dataset to run prediction for; will be ignored if --evaluate is test',
    )
    parser.add_argument('--tasks', dest='task_names', nargs='+', help='task names for prediction')
    parser.add_argument('--extra_metrics', nargs='+', default=[], help='include these additional metrics in reported results')
    parser.add_argument(
        '--devices',
        default=None,
        nargs='+',
        type=int,
        help='a list of devices that can be used for prediction. By default, all devices will be used.',
    )
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='.embeddings/', type=str, help='where to save embeddings.')
    parser.add_argument(
        '--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)'
    )
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite previously written predictions')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')

    parser.add_argument('--eval_dir', type=str, required=True, help='use this directory to store eval results')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the eval/test datasets')

    parser.add_argument(
        '--pred_languages',
        type=str,
        nargs='+',
        dest='pred_src_languages',
        help='Specify dataset source languages used during prediction for multilingual tasks',
    )
    parser.add_argument(
        '--pred_tgt_languages',
        type=str,
        nargs='+',
        help='Specify dataset target languages used during prediction for multilingual tasks',
    )

    parser.add_argument(
        '--main_metric_only', action='store_true', help='If True, we only calculate the deca score metric for each task.'
    )
    # If not None, these values will override the values saved in the trained model's config file
    parser.add_argument(
        '--val_batch_size',
        nargs='+',
        default=None,
        type=int,
        help='Batch size for validation corresponding to tasks in val tasks',
    )
    parser.add_argument(
        "--reduce_metrics",
        type=str,
        default='max',
        choices=['max', 'top_k'],
        help='How to calculate the metric when there are multiple outputs per input.'
        '`max` chooses the best set of generation hyperparameters and reports the metric for that.'
        '`top_k` chooses the best generation output per input, and uses that to output the metric. For example, combining this with the exact match metric gives what is commonly known as the top-k accuracy. Note that the output is meaningless if used with corpus-level metrics.',
    )

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate `num_outputs` outputs for each set of hyperparameters.
    parser.add_argument("--num_outputs", type=int, nargs='+', default=[1], help='number of sequences to output per input')
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.0], help="temperature of 0 implies greedy sampling")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        nargs='+',
        default=[1.0],
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[1.0], help='1.0 disables top-p filtering')
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')
    parser.add_argument("--num_beam_groups", type=int, nargs='+', default=[1], help='1 disables diverse beam seach')
    parser.add_argument("--diversity_penalty", type=float, nargs='+', default=[0.0], help='0 disables diverse beam seach')
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        nargs='+',
        default=[0],
        help='ngrams of this size cannot be repeated in the output. 0 disables it.',
    )
    parser.add_argument('--max_output_length', type=int, help='maximum output length for generation')
    parser.add_argument(
        '--min_output_length',
        type=int,
        help='maximum output length for generation; '
        'default is 3 for most multilingual models: BOS, language code, and one token. otherwise it is 2',
    )

    # These are used for confidence calibration
    parser.add_argument(
        '--calibrator_paths',
        type=str,
        nargs='+',
        default=None,
        help='Can be a list. If provided, each calibrator will be used to output confidence scores for each prediction.',
    )
    parser.add_argument(
        '--save_confidence_features',
        action='store_true',
        help='If provided, will be used to output confidence scores for each prediction.',
    )
    parser.add_argument(
        "--confidence_feature_path", type=str, default=None, help='A .pkl file to save confidence features in.'
    )
    parser.add_argument(
        "--mc_dropout_num",
        type=int,
        default=0,
        help='Number of samples to use for Monte Carlo (MC) dropout. 0 disables MC dropout.',
    )
    parser.add_argument(
        "--override_confidence_labels",
        type=str,
        default=None,
        help='If provided, examples with this gold answer are marked as 1, and others as 0. Useful for out-of-domain detection.',
    )

    parser.add_argument(
        '--database_dir', type=str, help='Path to folder containing all files (e.g. alias2qids, pretrained models for bootleg)'
    )

    parser.add_argument(
        "--mixed_precision",
        action='store_true',
        help='If True, will use mixed precision for prediction.'
        'This reduces memory consumption and is especially faster on GPUs like NVIDIA V100 and T4. May slightly change the generated output.',
    )
    parser.add_argument(
        '--one_output_per_line',
        action='store_true',
        help='If true, each of the `num_outputs` output will be written to a separate line, while other columns are duplicated to fill these extra lines.',
    )

    # TODO Update other tasks to use this argument too; so we can use predict for pure text generation (i.e. without reporting accuracy metrics)
    parser.add_argument(
        '--translate_no_answer',
        action='store_true',
        help='if true the provided dataset would not contain the answer (translated sentence)',
    )
    parser.add_argument(
        '--translate_example_split',
        action='store_true',
        help='split examples with multiple sentences into individual examples',
    )

    parser.add_argument(
        '--translate_return_raw_outputs',
        action='store_true',
        help='return raw translation as well as ones post-processed with alignment. this is useful for STS filtering.',
    )

    parser.add_argument('--plot_heatmaps', action='store_true', help='whether to plot cross-attention heatmaps')
    parser.add_argument(
        '--do_alignment',
        action='store_true',
        help='whether to replace tokens between quotation marks after translation with source values',
    )
    parser.add_argument(
        '--align_preserve_input_quotation',
        action='store_true',
        help='preserve quotation marks in the input. Useful if using alignment for semantic parsing or NLG',
    )
    parser.add_argument(
        '--align_remove_output_quotation',
        action='store_true',
        help='do not preserve quotation marks in the output. Useful if using alignment for semantic parsing or NLG',
    )
    parser.add_argument(
        '--align_span_symbol',
        type=str,
        help='The symbol we use to wrap spans of words in the input that need to be preserved in the output.',
    )

    parser.add_argument(
        '--e2e_dialogue_evaluation',
        action='store_true',
        help='Evaluate model on a dialogue dataset end-to-end; i.e. model predictions are used as input instead of gold',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_subtasks',
        nargs='+',
        type=str,
        help='Evaluate only on these subtasks when calculating e2e_dialogue_score; rg is not included by default',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_submetrics',
        nargs='+',
        type=str,
        help='Specify metrics to use for each of subtasks in e2e_dialogue_valid_subtasks.',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_subweights',
        nargs='+',
        type=float,
        help='Specify weights to use for each of subtasks in e2e_dialogue_valid_subtasks.',
    )

    parser.add_argument(
        '--model_parallel_hf',
        action='store_true',
        help='Use model parallelization by splitting model weights across available gpus',
    )


#
# def run(args, devices):
#     device = devices[0]
#
#     # TODO handle multiple languages
#     Model = getattr(models, args.model)
#     model, _ = Model.load(
#         args.path,
#         model_checkpoint_file=args.checkpoint_name,
#         args=args,
#         device='cpu',
#         tasks=args.tasks,
#         src_lang=args.pred_src_languages[0],
#         tgt_lang=args.pred_tgt_languages[0],
#     )
#     val_sets = prepare_data(args)
#
#     model.add_new_vocab_from_data(args.tasks)
#
#     logger.error('*******Start parallel hf********')
#     if args.model_parallel_hf:
#         # model.to('cpu')
#         parallelize(model.model, num_gpus=len(devices), fp16=args.mixed_precision, verbose='detail')
#     logger.error('*******Finish parallel hf********')
#
#     model = model.cuda()
#
#     iters = prepare_data_iterators(args, val_sets, model.numericalizer, device)
#
#     log_model_size(logger, model, args.model)
#
#     model.eval()
#
#     eval_dir = os.path.join(args.eval_dir, args.evaluate)
#     os.makedirs(eval_dir, exist_ok=True)
#
#     for index, (task, it, original_order) in enumerate(iters):
#         logger.info(task.name)
#
#         with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision):
#             validation_output = model.validate(
#                 it,
#                 task,
#                 eval_dir=eval_dir,
#                 original_order=original_order,
#                 disable_progbar=False,
#             )
#
#     return validation_output


def run(args, devices):

    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    tokens = tokenizer(
        "Kevin is <mask> man in the <mask>. Today He is working with his friend.",
        return_tensors="pt",
    )

    for t in tokens:
        if torch.is_tensor(tokens[t]):
            tokens[t] = tokens[t].cuda()

    logger.error('*******Start parallel hf********')
    if args.model_parallel_hf:
        # model.to('cpu')
        parallelize(model, num_gpus=len(devices), fp16=args.mixed_precision, verbose='detail')
    logger.error('*******Finish parallel hf********')

    model = model.cuda()

    log_model_size(logger, model, args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parse_argv(parser)
    args = parser.parse_args()

    load_config_json(args)
    check_and_update_generation_args(args)
    check_args(args)
    set_default_values(args)

    set_seed(args)
    args.tasks = list(get_tasks(args.task_names, args).values())

    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')
    devices = get_devices(args.devices)

    logger.info(f'Multi device generation on: {devices}')
    run(args, devices)
