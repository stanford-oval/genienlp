import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

def parse_argv(parser):
    
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--subsample', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda')


def main(args):
    
    model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")
    
    # model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    # model = SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")
    # model = SentenceTransformer("distilbert-multilingual-nli-stsb-quora-ranking")
    # model = SentenceTransformer("LaBSE")
    
    # src_sentences = ['I am here', 'I am here', 'I am here', 'I am here']
    # trg_sentences = ['I am not there', 'He is here', 'I am there', 'I am here']
    # labse_trg_sentences = ['Non sono li', 'Lui è qui', 'Ci sono', 'Sono qui']
    # labels = [1, 0.5, 0, 1]
    
    if args.device == 'cuda' and torch.cuda.is_available():
        model.cuda()
    
    src_sentences = []
    trg_sentences = []
    
    with open(args.input_file, 'r') as fin:
        for i, line in enumerate(fin):
            row = list(map(lambda part: part.strip(), line.split('\t')))
            src_sentences.append(row[0])
            trg_sentences.append(row[1])
            
            if args.subsample != -1 and i >= args.subsample:
                break
        
    embeddings1 = model.encode(src_sentences, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    embeddings2 = model.encode(trg_sentences, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    
    with open(args.output_file, 'w') as fout:
        for src, tgt, score in zip(src_sentences, trg_sentences, cosine_scores):
            fout.write('\t'.join([src, tgt, '{:0.4f}'.format(score)]) + '\n')
