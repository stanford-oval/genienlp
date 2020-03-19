#
# Copyright (c) 2019, The Board of Trustees of the Leland Stanford Junior University
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


import logging
from .base_task import BaseTask

logger = logging.getLogger(__name__)


class TaskRegistry:
    """
    A container for all the tasks supported by the library
    """

    def __init__(self):
        self.store = dict()

    def __setitem__(self, name, cls):
        if name in self.store:
            raise ValueError(f'Duplicate task {name}')
        self.store[name] = cls

    def __getitem__(self, name):
        if name not in self.store:
            logger.warning(f'Unrecognized task {name}, using generic code')
            self.store[name] = BaseTask
        return self.store[name]

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store)


_registry = TaskRegistry()
_registry['generic'] = BaseTask


def task_name_to_cls_name(name):
    return name.split('.')[0]


def register_task(name):
    def decorator(cls):
        _registry[name] = cls
        return cls

    return decorator


def get_tasks(names, args):
    tasks = {
        name: (_registry[task_name_to_cls_name(name)])(name, args) for name in names
    }
    return [tasks[name] for name in names]
