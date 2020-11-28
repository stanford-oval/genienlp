import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

def parse_argv(parser):
    
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--subsample', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='xlm-r-distilroberta-base-paraphrase-v1',
                        help='List of available models: https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/')


def main(args):
    
    model = SentenceTransformer(args.model_name)
    
    if args.device == 'cuda' and torch.cuda.is_available():
        model.cuda()
    
    ids = []
    src_sentences = []
    trg_sentences = []
    programs = []
    
    with open(args.input_file, 'r') as fin:
        for i, line in enumerate(fin):
            row = list(map(lambda part: part.strip(), line.split('\t')))
            ids.append(row[0])
            src_sentences.append(row[1])
            trg_sentences.append(row[2])
            programs.append(row[3])
            
            if args.subsample != -1 and i >= args.subsample:
                break
        
    embeddings1 = model.encode(src_sentences, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    embeddings2 = model.encode(trg_sentences, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    
    with open(args.output_file, 'w') as fout:
        for id_, src, tgt, score, program, in zip(ids, src_sentences, trg_sentences, cosine_scores, programs):
            fout.write('\t'.join([id_, src, tgt, '{:0.4f}'.format(score), program]) + '\n')
