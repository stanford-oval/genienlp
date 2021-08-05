#
# Copyright (c) 2021, Salesforce, Inc.
#                     The Board of Trustees of the Leland Stanford Junior University
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

import kfserving

from .server import Server, init
from .util import log_model_size

logger = logging.getLogger(__name__)


class KFModelServer(kfserving.KFModel):
    def __init__(self, name, args, numericalizer, model, device, confidence_estimators, estimator_filenames, ned_model):
        super().__init__(name)
        self.server = Server(args, numericalizer, model, device, confidence_estimators, estimator_filenames, ned_model)

    def load(self):
        log_model_size(logger, self.server.model, self.server.args.model)
        self.server.model.to(self.server.device)
        self.server.model.eval()
        self.ready = True

    def predict(self, request):
        results = self.server.handle_request(request)
        return {"predictions": results}


def main(args):
    model, device, confidence_estimators, estimator_filenames, ned_model = init(args)
    model_server = KFModelServer(
        args.inference_name, args, model.numericalizer, model, device, confidence_estimators, estimator_filenames, ned_model
    )
    model_server.load()
    kfserving.KFServer(workers=1).start([model_server])
