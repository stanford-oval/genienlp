#
# Copyright (c) 2018, Salesforce, Inc.
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

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


class Multiprocess():

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        self.world_size = args.world_size

        if os.path.isfile(args.dist_sync_file):
            os.remove(args.dist_sync_file)

    def run(self, runtime_args):
        self.start(runtime_args)
        self.join()

    def start(self, runtime_args):
        self.processes = []
        for rank in range(self.world_size):
            self.processes.append(Process(target=self.init_process, args=(rank, self.fn, self.args, runtime_args)))
            self.processes[-1].start()

    def init_process(self, rank, fn, args, runtime_args):
        torch.distributed.init_process_group(world_size=self.world_size, 
                                             init_method='file://'+args.dist_sync_file, 
                                             backend=args.backend, 
                                             rank=rank)
        fn(args, runtime_args, rank, self.world_size)
  
    def join(self):
        for p in self.processes:
            p.join()



