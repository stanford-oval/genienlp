import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
from copy import deepcopy as copy_deep
from copy import copy as copy
from modules.matrixBLEU import mBLEU
from modules.utils import CUDA_wrapper
from collections import Counter
from modules.utils import LongTensor, FloatTensor
from functools import reduce
from modules.utils import CUDA_wrapper
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Reslicer:
    def __init__(self, max_lenght):
        """
        This functor is used to prevent empty reslice
        of index selecting when it appears to be zero
        """
        self.max_l = max_lenght

    def __call__(self, x):
        return self.max_l - x

def ngrams_product(A, n):
    """
    A-is probability matrix
    [batch x length_candidate_translation x reference_len]
    third dimension is reference's words in order of appearance in reference
    n - states for n-grams
    Output: [batch, (length_candidate_translation-n+1) x (reference_len-n+1)]
    """
    max_l = min(A.size()[1:])
    reslicer = Reslicer(max_l)
    if reslicer(n-1) <= 0:
        return None
    cur = A[:, :reslicer(n-1), :reslicer(n-1)].clone()
    for i in range(1, n):
        mul = A[:, i:reslicer(n-1-i), i:reslicer(n-1-i)]
        cur = cur * mul
    return cur

def get_selected_matrices(probs, references, dim=1):
    """
    batched index select
    probs - is a matrix
    references - is index
    dim - is dimention of element of the batch
    """
    # NOTE for loop in index select. Found only this way to do this.
    # It seems that it could be optimized via batched version of index_select
    # but there is no batched_index_select in pytorch for now
    return torch.cat([torch.index_select(a, dim, Variable(LongTensor(i))).unsqueeze(0)\
                            for a, i in zip(probs, references)])


def ngram_ref_counts(reference, lengths, n):
    """
    For each position counts n-grams equal to n-gram to this position
    reference - matrix sequences of id's from vocabulary.[batch, ref len]
    NOTE reference should be padded with some special ids
    At least one value in length must be equal reference.shape[1]
    output: counts n-grams for each start position padded with zeros
    """
    res = []
    max_len = max(lengths)
    if max_len - n + 1 <= 0:
        return None
    for r, l in zip(reference, lengths):
        picked = set() # we only take into account first appearance of n-gram
        #             (which contains its count of occurrence)
        current_length = l - n + 1
        cnt = Counter([tuple([r[i + j] for j in range(n)]) \
                        for i in range(current_length)])
        occurrence = []
        for i in range(current_length):
            n_gram = tuple([r[i + j] for j in range(n)])
            val = 0
            if not n_gram in picked:
                val = cnt[n_gram]
                picked.add(n_gram)
            occurrence.append(val)
        padding = [1 for _ in range(max_len - l if current_length > 0\
                                                else max_len - n+ 1)]
        res.append(occurrence + padding)
    return Variable(FloatTensor(res), requires_grad=False)

def calculate_overlap(p, r, n, lengths):
    """
    p - probability tensor [b x len_x x reference_length]
    r - references, tensor [b x len_y]
    contains word's ids for each reference in batch
    n - n-gram
    lenghts - lengths of each reference in batch
    """
    A = ngrams_product(get_selected_matrices(p, r), n)
    r_cnt = ngram_ref_counts(r, lengths, n)
    if A is None or r_cnt is None:
        return CUDA_wrapper(torch.zeros(p.shape[0]))
    r_cnt = r_cnt[:, None]
    A_div = -A + torch.sum(A, 1, keepdim=True) + 1
    second_arg = r_cnt / A_div
    term = torch.min(A, A * second_arg)
    return torch.sum(torch.sum(term, 2), 1)

def bleu(p, r, translation_lengths, reference_lengths, max_order=4, smooth=False):
    """
    p - matrix with probabilityes
    r - reference batch
    reference_lengths - lengths of the references
    max_order - max order of n-gram
    smooth - smooth calculation of precisions
    translation_lengths - torch tensor
    """
    overlaps_list = []
    translation_length = sum(translation_lengths)
    reference_length = sum(reference_lengths)
    for n in range(1, max_order + 1):
        overlaps_list.append(calculate_overlap(p, r, n, reference_lengths))
    overlaps = CUDA_wrapper(torch.stack(overlaps_list))
    matches_by_order = torch.sum(overlaps, 1)
    possible_matches_by_order = CUDA_wrapper(torch.zeros(max_order))
    for n in range(1, max_order + 1):
        cur_pm = translation_lengths.float() - n + 1
        mask = cur_pm > 0
        cur_pm *= mask.float()
        possible_matches_by_order[n - 1] = torch.sum(cur_pm)
    precisions = Variable(FloatTensor([0] * max_order))
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1) /\
                                            (possible_matches_by_order[i] + 1)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = matches_by_order[i] /\
                                            possible_matches_by_order[i]
            else:
                precisions[i] = Variable(FloatTensor([0]))
    if torch.min(precisions[:max_order]).item() > 0:
        p_log_sum = sum([(1. / max_order) * torch.log(p) for p in precisions])
        geo_mean = torch.exp(p_log_sum)
    else:
        geo_mean = torch.pow(\
                        reduce(lambda x, y: x*y, precisions), 1./max_order)
        eprint('WARNING: some precision(s) is zero')
    ratio = float(translation_length) / reference_length
    if ratio > 1.0:
        bp = 1.0
    else:
        THRESHOLD_RATIO = 1E-1
        MIN_BP = 1E-2
        if ratio > THRESHOLD_RATIO:
            bp = np.exp(1 - 1. / ratio)
        else:
            bp = MIN_BP
    bleu = -geo_mean * bp
    return bleu, precisions
