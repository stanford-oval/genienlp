import torch
from torch.nn import functional
from torch.autograd import Variable
import numpy as np
import os
from functools import reduce
from copy import deepcopy as copy
import time
from modules.utils import CUDA_wrapper
from modules.utils import SoftmaxWithTemperature
from modules.utils import fill_eye_diag

class mBLEU:
    def __init__(self, max_order=4, softmax_temperature=0.001, T_argmax=True,\
                std_temp=False):
        """class implementing straightforwad matrix BLEU computation"""
        self.max_order = max_order
        self.T_argmax = T_argmax
        self.sm = SoftmaxWithTemperature(softmax_temperature)
        self.softmax_regular = torch.nn.Softmax()
        self.std_temp = std_temp

    def __call__(self, R, T, reference_corpus_lens, translation_corpus_lens):
        """
        T[b x t x v]
        R[b x r]
        reference_corpus_lens - list, len=b
        translation_corpus_lens - list, len=b
        """
        max_order = self.max_order
        shapeR = R.data.shape
        shapeT = T.data.shape
        translation_length = sum(translation_corpus_lens)
        reference_length = sum(reference_corpus_lens)
        if self.T_argmax:
            cur_temperature = None
            if self.std_temp:
                cur_temperature = T.std()
                if (np.random.rand(1)[0] > 0.99):
                    print(cur_temperature)
            T = self.sm(T.contiguous().view(-1, shapeT[2]),\
                                    temperature=cur_temperature).view(shapeT)
        TR = T.bmm(R.transpose(1, 2))
        TT = T.bmm(T.transpose(1, 2))
        # TT = fill_eye_diag(TT)

        reference_len = sum(reference_corpus_lens)
        tanslation_len = sum(translation_corpus_lens)
        matches_by_order = [CUDA_wrapper(Variable(torch.FloatTensor([0])))\
                                        for i in range(max_order)]
        cur_t = TT
        cur_tr = TR
        all_t = [torch.sum(cur_t, 1)]
        all_tr = [torch.sum(cur_tr, 2)]
        def overlapper(t, tr):
            SMOOTH_CONST = 1E-10
            return torch.sum((torch.min(t, tr) + SMOOTH_CONST) / torch.max(\
                (t + SMOOTH_CONST),CUDA_wrapper(Variable(\
                                                torch.FloatTensor([1])))), 1)
        overlap = overlapper(all_t[-1], all_tr[-1])
        matches_by_order[0] = torch.sum(overlap)
        possible_matches_by_order = [
                                CUDA_wrapper(Variable(torch.FloatTensor([0])))\
                                for i in range(max_order)\
                                    ]
        def update_possible_matches(possible_matches_by_order,\
                                                translation_corpus_lens, order):
            for transl_len in translation_corpus_lens:
                possible_matches = transl_len - order
                if possible_matches > 0:
                    possible_matches_by_order[order] += possible_matches
        update_possible_matches(possible_matches_by_order,\
                                                translation_corpus_lens, 0)
        for order in range(1, min(max_order, shapeT[1], shapeR[1])):
            cur_t = TT[:, order:, order:] * cur_t[:, :-1, :-1]
            all_t.append(torch.sum(cur_t, 1))
            cur_tr = TR[:, order:, order:] * cur_tr[:, :-1, :-1]
            all_tr.append(torch.sum(cur_tr, 2))
            overlap = overlapper(all_t[-1], all_tr[-1])
            matches_by_order[order] = torch.sum(overlap)
            update_possible_matches(possible_matches_by_order,\
                                            translation_corpus_lens, order)

        precisions = [CUDA_wrapper(Variable(torch.FloatTensor([0])))\
                                                    for i in range(max_order)]
        for i in range(0, max_order):
            if possible_matches_by_order[i].data[0] > 0:
                if i > 0:
                    precisions[i] = ((matches_by_order[i].float() + 1)\
                                        /( possible_matches_by_order[i] + 1))
                else:
                    precisions[i] = (matches_by_order[i].float()\
                                        /possible_matches_by_order[i])
            else:
                precisions[i] = CUDA_wrapper(Variable(torch.FloatTensor([0])))
        if torch.min(torch.stack(precisions)).data[0] > 1E-3:
            p_log_sum = sum([(1. / max_order) * torch.log(p)\
                                                        for p in precisions])
            geo_mean = torch.exp(p_log_sum)
        else:
            geo_mean = torch.pow(\
                            reduce(lambda x, y: x*y, precisions), 1./max_order)
        ratio = float(translation_length) / reference_length
        if ratio > 1.0:
            bp = 1.
        else:
            THRESHOLD_RATIO = 1E-1
            MIN_BP = 1E-2
            if ratio > THRESHOLD_RATIO:
                bp = np.exp(1 - 1. / ratio)
            else:
                bp = MIN_BP
        bleu = -geo_mean * bp
        return bleu, precisions
