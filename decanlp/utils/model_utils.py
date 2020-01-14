#!/usr/bin/python3
#
# Copyright 2019 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Mehrad Moradshahi <mehrad@cs.stanford.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ..util import get_trainable_params, log_model_size

def init_model(args, field, logger, device, model_name=None):
    if not model_name:
        model_name = args.model
    logger.info(f'Initializing {model_name}')
    Model = getattr(models, model_name)
    model = Model(field, args)
    params = get_trainable_params(model)
    log_model_size(logger, model, model_name)

    model.to(device)

    model.params = params
    return model

from decanlp import models # break import loop by importing models at the bottom of the script