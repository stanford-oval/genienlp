from argparse import ArgumentParser
import random
import os

def load_file(file_path):
    lines = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    return lines

def load_examples(file_path):
    lines = load_file(file_path)
    examples = []
    for i in range(len(lines)):
        if lines[i].startswith('(example'):
            examples.append((lines[i+1][12:-2], lines[i+2][11:-2], lines[i+4][:])) # natural, synthetic, target
    return examples

def write_to_folder(train_examples, eval_examples, test_examples, folder_path, dev_portion, subset='natural'):
    os.makedirs(os.path.join(folder_path, subset), exist_ok=True)
    train_file = open(os.path.join(folder_path, subset, 'train.tsv'), 'w')
    eval_file = open(os.path.join(folder_path, subset, 'eval.tsv'), 'w')
    test_file = open(os.path.join(folder_path, subset, 'test.tsv'), 'w')

    i = 0
    for e in train_examples:
        if subset == 'natural':
            line = 'R'+str(i)+'\t'+e[0]+'\t'+e[2]+'\n'
        else:
            line = 'RS'+str(i)+'\t'+e[1]+'\t'+e[2]+'\n'
        train_file.write(line)
        i += 1

    i = 0
    for e in eval_examples:
        line = 'R'+str(i)+'\t'+e[0]+'\t'+e[2]+'\n' # eval is always natural
        eval_file.write(line)
        i += 1

    i = 0
    for e in test_examples:
        line = 'R'+str(i)+'\t'+e[0]+'\t'+e[2]+'\n'
        test_file.write(line)
        i += 1

    train_file.close()
    eval_file.close()
    test_file.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('input_folder', type=str, help='The path to the input file.')
    parser.add_argument('output_folder', type=str, help='The folder of synthetic and natural outputs.')
    parser.add_argument('--dev_portion', type=float, default=0.2, help='The portion of training set that should be used for development.')
    parser.add_argument('--domain_name', type=str, help='The name of the overnight domain.')

    args = parser.parse_args()

    train_eval_examples = load_examples(os.path.join(args.input_folder, args.domain_name+'.paraphrases.train.examples'))
    test_examples = load_examples(os.path.join(args.input_folder, args.domain_name+'.paraphrases.test.examples'))

    train_examples = []
    eval_examples = []
    for e in train_eval_examples:
        r = random.random()
        if r < args.dev_portion:
            eval_examples.append(e)
        else:
            train_examples.append(e)


    write_to_folder(train_examples, eval_examples, test_examples, os.path.join(args.output_folder, 'overnight-'+args.domain_name), args.dev_portion, 'synthetic')
    write_to_folder(train_examples, eval_examples, test_examples, os.path.join(args.output_folder, 'overnight-'+args.domain_name), args.dev_portion, 'natural')
    

if __name__ == '__main__':
    main()