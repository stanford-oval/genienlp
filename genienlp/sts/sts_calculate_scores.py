#
# Copyright (c) 2020-2021 The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
from transquest.algo.sentence_level.siamesetransquest.run_model import SiameseTransQuestModel


def parse_argv(parser):

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--subsample', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--embeddings', type=str, default='.embeddings')
    parser.add_argument('--model_type', type=str, choices=['st', 'tq-mono', 'tq-siam'])
    parser.add_argument(
        '--model_name',
        type=str,
        help='List of available sts models: https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/'
        'List of available transquest models: https://tharindu.co.uk/TransQuest/models/sentence_level_pretrained/',
    )


def main(args):
    use_cuda = args.device == 'cuda' and torch.cuda.is_available()

    if args.model_type == 'tq-mono':
        model = MonoTransQuestModel("xlmroberta", args.model_name, num_labels=1, use_cuda=use_cuda, cache_dir=args.embeddings)
    elif args.model_type == 'tq-siam':
        # cache_dir is not supported yet!
        model = SiameseTransQuestModel(args.model_name)
    else:
        model = SentenceTransformer(args.model_name, device=args.device, cache_folder=args.embeddings)

    ids = []
    src_sentences = []
    tgt_sentences = []
    programs = []

    with open(args.input_file, 'r') as fin:
        for i, line in enumerate(fin):
            row = list(map(lambda part: part.strip(), line.split('\t')))
            ids.append(row[0])
            src_sentences.append(row[1])
            tgt_sentences.append(row[2])
            if len(row) > 3:
                programs.append(row[3])

            if args.subsample != -1 and i >= args.subsample:
                break

    if args.model_type == 'tq-mono':
        cosine_scores, _ = model.predict(list(map(list, zip(src_sentences, tgt_sentences))))
    elif args.model_type == 'tq-siam':
        cosine_scores = model.predict(list(zip(src_sentences, tgt_sentences)))
    else:
        embeddings1 = model.encode(src_sentences, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
        embeddings2 = model.encode(tgt_sentences, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

    with open(args.output_file, 'w') as fout:
        for i in range(len(ids)):
            id_, src, tgt, score = ids[i], src_sentences[i], tgt_sentences[i], cosine_scores[i]
            prog = None
            if programs:
                prog = programs[i]
            fout.write('\t'.join([id_, src, tgt, '{:0.4f}'.format(score), prog if prog else '']) + '\n')
