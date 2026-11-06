# Derived from the incremental learning codebase of Zheng et al. (ACL 2024),
# https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm
# Modifications copyright (c) 2026 Keuntae Kim.
# See NOTICE for details.

"""
Entry point for all incremental learning experiments.

Example:
    python main_CL.py --exp_prefix demo \
        --cfg ./config/CIL/generative_backbones/clinc150_task15/SEQ_full.yaml \
        --backbone EleutherAI/pythia-410m-deduped --classifier None --training_epochs 5
"""
import os

# Must be set before torch / tokenizers are initialized.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
if not torch.cuda.is_initialized():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import gc
import time
import random

import numpy as np
import wandb
from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size

from utils.config import get_params
from utils.factory import get_model
from utils.dataset import get_dataset
from utils.logger import init_experiment

# Backbones for which the automatic batch-size search is disabled, so that the
# requested batch size is used as is.
FIXED_BATCH_SIZE_BACKBONES = ('meta-llama/Llama-3.2-1B',)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def release_memory(accelerator):
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def init_wandb(accelerator, params, max_retry=5):
    """Initialize the wandb tracker, retrying on transient connection errors."""
    if 'default' in params.exp_prefix:
        mode = 'disabled'
    else:
        mode = 'online' if params.is_wandb else 'offline'
    for _ in range(max_retry):
        try:
            accelerator.init_trackers(
                project_name=params.wandb_project,
                init_kwargs={
                    'wandb': {
                        'entity': params.wandb_entity,
                        'name': params.wandb_name,
                        'config': vars(params),
                        'settings': wandb.Settings(start_method='fork'),
                        'mode': mode,
                    }
                }
            )
            return
        except Exception as e:
            print(str(e))
            print('Retrying to connect to wandb...')
            time.sleep(120)


def main_cl(params):
    accelerator = Accelerator(log_with='wandb')

    if params.seed:
        set_random_seed(params.seed)

    logger = init_experiment(params, logger_filename=params.logger_filename)
    logger.info(params.__dict__)

    init_wandb(accelerator, params)

    CL_dataset = get_dataset(params)

    def run():
        release_memory(accelerator)
        model = get_model(params, CL_dataset, accelerator)
        model.incremental_training()
        model.finish_training()

    if params.backbone in FIXED_BATCH_SIZE_BACKBONES:
        run()
    else:
        @find_executable_batch_size(starting_batch_size=params.batch_size)
        def run_with_batch_size_search(batch_size):
            params.__setattr__('batch_size', batch_size)
            run()

        run_with_batch_size_search()


if __name__ == '__main__':
    main_cl(get_params())
