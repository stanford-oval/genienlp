import numpy as np

def parse_argv(parser):
    
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--subsample', type=int, default=-1)
    parser.add_argument('--filtering_metric', type=str, default='mean', choices=['mean', 'mean+std', 'all'])


def main(args):
    
    all_scores = []
    
    with open(args.input_file, 'r') as fin:
        for line in fin:
            parts = list(map(lambda p: p.strip(), line.split('\t')))
            id_, orig_sent, para_sent, score, program = parts
            all_scores.append(score)

    all_scores = np.array(all_scores, dtype=float)
    scores_mean = np.mean(all_scores)
    scoers_std = np.std(all_scores)
    
    if args.filtering_metric == 'mean':
        accepted_ids = all_scores >= scores_mean
    elif args.filtering_metric == 'mean+std':
        accepted_ids = all_scores >= (scores_mean + scoers_std)
    # accept all
    else:
        accepted_ids = all_scores >= 0.0
    
    with open(args.input_file, 'r') as fin, open(args.output_file, 'w') as fout:
        for i, line in enumerate(fin):
            if accepted_ids[i]:
                parts = list(map(lambda p: p.strip(), line.split('\t')))
                id_, orig_sent, para_sent, score, program = parts
                fout.write('\t'.join(['P' + id_, para_sent, program]) + '\n')
