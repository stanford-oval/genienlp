from decanlp.util import get_trainable_params, count_params
from decanlp.multiprocess import Multiprocess, DistributedDataParallel

def init_model(args, field, logger, world_size, device, model_name=None):
    if not model_name:
        model_name = args.model
    logger.info(f'Initializing {model_name}')
    Model = getattr(models, model_name)
    model = Model(field, args)
    params = get_trainable_params(model)
    num_param = count_params(params)
    logger.info(f'{args.model} has {num_param:,} trainable parameters')

    model.to(device)
    if world_size > 1:
        logger.info(f'Wrapping model for distributed')
        model = DistributedDataParallel(model)

    model.params = params
    return model

from decanlp import models # break import loop by importing models at the bottom of the script