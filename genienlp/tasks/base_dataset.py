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


import os
import tarfile
import urllib
import zipfile
from typing import NamedTuple, Union

import requests
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields.

    Attributes:
        sort_key (callable): A key to use for sorting dataset examples for batching
            together examples with similar lengths to minimize padding.
        examples (list(Example)): The examples in this dataset.
    """

    sort_key = None

    def __init__(self, examples, filter_pred=None, **kwargs):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            examples: List of Examples.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        self.examples = examples

    @classmethod
    def splits(cls, root='.data', train=None, validation=None, test=None, **kwargs):
        """Create Dataset objects for multiple splits of a dataset.

        Arguments:
            root (str): Root dataset storage directory. Default is '.data'.
            train (str): Suffix to add to path for the train set, or None for no
                train set. Default is None.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            Remaining keyword arguments: Passed to the constructor of the
                Dataset (sub)class being used.

        Returns:
            split_datasets (tuple(Dataset)): Datasets for train, validation, and
                test splits in that order, if provided.
        """
        path = cls.download(root)
        train_data = None if train is None else cls(os.path.join(path, train), **kwargs)
        val_data = None if validation is None else cls(os.path.join(path, validation), **kwargs)
        test_data = None if test is None else cls(os.path.join(path, test), **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __repr__(self):
        if self.examples is not None:
            return 'Dataset(' + self.examples.__repr__() + ')'
        else:
            return 'Dataset()'

    @classmethod
    def download(cls, root, check=None):
        """Download and unzip an online archive (.zip, .gz, or .tgz).

        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.

        Returns:
            dataset_path (str): Path to extracted dataset.
        """
        path = os.path.join(root, cls.name)
        check = path if check is None else check
        if not os.path.isdir(check):
            for url in cls.urls:
                if isinstance(url, tuple):
                    url, filename = url
                else:
                    filename = os.path.basename(url)
                zpath = os.path.join(path, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print('downloading {}'.format(filename))
                    download_from_url(url, zpath)
                ext = os.path.splitext(filename)[-1]
                if ext == '.zip':
                    with zipfile.ZipFile(zpath, 'r') as zfile:
                        print('extracting')
                        zfile.extractall(path)
                elif ext in ['.gz', '.tgz']:
                    with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        def is_within_directory(directory, target):
                            
                            abs_directory = os.path.abspath(directory)
                            abs_target = os.path.abspath(target)
                        
                            prefix = os.path.commonprefix([abs_directory, abs_target])
                            
                            return prefix == abs_directory
                        
                        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        
                            for member in tar.getmembers():
                                member_path = os.path.join(path, member.name)
                                if not is_within_directory(path, member_path):
                                    raise Exception("Attempted Path Traversal in Tar File")
                        
                            tar.extractall(path, members, numeric_owner=numeric_owner) 
                            
                        
                        safe_extract(tar, path=path, members=dirs)
                elif ext in ['.bz2', '.tar']:
                    with tarfile.open(zpath) as tar:
                        dirs = [member for member in tar.getmembers()]
                        def is_within_directory(directory, target):
                            
                            abs_directory = os.path.abspath(directory)
                            abs_target = os.path.abspath(target)
                        
                            prefix = os.path.commonprefix([abs_directory, abs_target])
                            
                            return prefix == abs_directory
                        
                        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        
                            for member in tar.getmembers():
                                member_path = os.path.join(path, member.name)
                                if not is_within_directory(path, member_path):
                                    raise Exception("Attempted Path Traversal in Tar File")
                        
                            tar.extractall(path, members, numeric_owner=numeric_owner) 
                            
                        
                        safe_extract(tar, path=path, members=dirs)

        return os.path.join(path, cls.dirname)


class Split(NamedTuple):
    train: Union[Dataset, str] = None
    eval: Union[Dataset, str] = None
    test: Union[Dataset, str] = None
    aux: Union[Dataset, str] = None


def interleave_keys(a, b):
    """Interleave bits from two sort keys to form a joint sort key.

    Examples that are similar in both of the provided keys will have similar
    values for the key defined by this function. Useful for tasks with two
    text fields like machine translation or natural language inference.
    """

    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])

    return int(''.join(interleave(format(x, '016b') for x in (a, b))), base=2)


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        try:
            return urllib.request.urlretrieve(url, path)
        except Exception:
            res = requests.get(url)
            with open(path, 'wb') as out:
                out.write(res.content)
    print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
