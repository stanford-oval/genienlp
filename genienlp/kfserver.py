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

# Part of this code was adopted from kuebflow
# Copyright 2020 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Union

import kfserving
import tornado
from kfserving.kfserver import (
    DEFAULT_GRPC_PORT,
    DEFAULT_HTTP_PORT,
    DEFAULT_MAX_BUFFER_SIZE,
    ExplainHandler,
    HealthHandler,
    KFModelRepository,
    ListHandler,
    LivenessHandler,
    LoadHandler,
    PredictHandler,
    UnloadHandler,
)

from .server import Server, init
from .util import log_model_size

logger = logging.getLogger(__name__)


# Overwrites write method to prevent transforming non-ascii characters into unicode escape characters
# See https://github.com/tornadoweb/tornado/issues/3070 for more details
class PredictHandlerV2(PredictHandler):
    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)

    def write(self, chunk: Union[str, bytes, dict]) -> None:
        """Writes the given chunk to the output buffer."""
        if self._finished:
            raise RuntimeError("Cannot write() after finish()")
        if not isinstance(chunk, (bytes, tornado.util.unicode_type, dict)):
            message = "write() only accepts bytes, unicode, and dict objects"
            if isinstance(chunk, list):
                message += (
                    ". Lists not accepted for security reasons; see "
                    + "http://www.tornadoweb.org/en/stable/web.html#tornado.web.RequestHandler.write"  # noqa: E501
                )
            raise TypeError(message)
        if isinstance(chunk, dict):
            chunk = json.dumps(chunk, ensure_ascii=False).replace("</", "<\\/")
            self.set_header("Content-Type", "application/json; charset=UTF-8")
        chunk = tornado.escape.utf8(chunk)
        self._write_buffer.append(chunk)


# overwrites create_application method to register PredictHandlerV2 instead of PredictHandler for prediction
class KFServer(kfserving.KFServer):
    def __init__(self, http_port, grpc_port, max_buffer_size, workers, registered_models):
        super().__init__(http_port, grpc_port, max_buffer_size, workers, registered_models)

    def create_application(self):
        return tornado.web.Application(
            [
                # Server Liveness API returns 200 if server is alive.
                (r"/", LivenessHandler),
                (r"/v2/health/live", LivenessHandler),
                (r"/v1/models", ListHandler, dict(models=self.registered_models)),
                (r"/v2/models", ListHandler, dict(models=self.registered_models)),
                # Model Health API returns 200 if model is ready to serve.
                (r"/v1/models/([a-zA-Z0-9_-]+)", HealthHandler, dict(models=self.registered_models)),
                (r"/v2/models/([a-zA-Z0-9_-]+)/status", HealthHandler, dict(models=self.registered_models)),
                (r"/v1/models/([a-zA-Z0-9_-]+):predict", PredictHandlerV2, dict(models=self.registered_models)),
                (r"/v2/models/([a-zA-Z0-9_-]+)/infer", PredictHandlerV2, dict(models=self.registered_models)),
                (r"/v1/models/([a-zA-Z0-9_-]+):explain", ExplainHandler, dict(models=self.registered_models)),
                (r"/v2/models/([a-zA-Z0-9_-]+)/explain", ExplainHandler, dict(models=self.registered_models)),
                (r"/v2/repository/models/([a-zA-Z0-9_-]+)/load", LoadHandler, dict(models=self.registered_models)),
                (r"/v2/repository/models/([a-zA-Z0-9_-]+)/unload", UnloadHandler, dict(models=self.registered_models)),
            ]
        )


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
    KFServer(
        http_port=DEFAULT_HTTP_PORT,
        grpc_port=DEFAULT_GRPC_PORT,
        max_buffer_size=DEFAULT_MAX_BUFFER_SIZE,
        workers=1,
        registered_models=KFModelRepository(),
    ).start([model_server])
