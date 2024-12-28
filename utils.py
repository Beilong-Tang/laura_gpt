import datetime
import os
import logging
import yaml
import random
import numpy as np
import torch

from argparse import Namespace


def init(module, config, *args, **kwargs):
    return getattr(module, config["type"])(*args, **kwargs, **config["args"])


def setup_logger(args: Namespace, rank: int, out=True):
    if out:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(
            args.log, os.path.basename(args.config).replace(".yaml", "")
        )
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
            handlers=[
                logging.FileHandler(f"{log_dir}/{now}.log"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
            handlers=[logging.StreamHandler()],
        )
    logger = logging.getLogger()
    logger.info("logger initialized")
    return Logger(logger, rank)


def update_args(args: Namespace, config_file_path: str):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        args.__setattr__(k, v)
    return args


class Logger:
    def __init__(self, log: logging.Logger, rank: int):
        self.log = log
        self.rank = rank

    def info(self, msg: str):
        if self.rank == 0:
            self.log.info(msg)

    def debug(self, msg: str):
        if self.rank == 0:
            self.log.debug(msg)
        pass

    def warning(self, msg: str):
        self.log.warning(f"rank {self.rank} - {msg}")
        pass

    def error(self, msg: str):
        self.log.error(f"rank {self.rank} - {msg}")
        pass

    def critical(self, msg: str):
        self.log.critical(f"rank {self.rank} - {msg}")

        pass


def setup_seed(seed, rank):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED


class AttrDict(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None
