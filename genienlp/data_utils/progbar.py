#
# Copyright (c) 2020 The Board of Trustees of the Leland Stanford Junior University
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

import math
import sys

from tqdm import tqdm


class LogFriendlyProgressBar:
    def __init__(self, iterable, desc, total, step: int = 1):
        """
        Inputs:
            step: Will print progress every `step` %
        """
        self._desc = desc
        self._i = 0
        self._N = total
        self.step = step
        self._progress = 0
        self._iterable = iterable
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self._iterable)
        return self

    def __next__(self):
        value = next(self._iterator)
        self._i += 1
        progress = math.floor(self._i * 100 / self._N)
        if progress > self._progress and progress % self.step == 0:
            if self._desc:
                print(f'{self._desc} - progress: {progress}%', file=sys.stderr)
            else:
                print(f'Progress: {progress}%', file=sys.stderr)
            sys.stderr.flush()
            self._progress = progress
        return value

    def close(self):
        return


def progress_bar(iterable, desc=None, total=None, disable=False):
    if disable:
        return iterable
    if total is None:
        if not hasattr(iterable, '__len__'):
            return iterable
        total = len(iterable)
    if sys.stderr.isatty():
        return tqdm(iterable, desc=desc, total=total)
    else:
        return LogFriendlyProgressBar(iterable, desc=desc, total=total, step=5)


def prange(*args, desc=None, disable=False):
    return progress_bar(range(*args), desc=desc, disable=disable)
