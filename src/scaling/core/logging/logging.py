import copy
import json
import logging
import os
import socket
from typing import Any, Optional

import wandb
from torch.utils.tensorboard import SummaryWriter

from ..config import BaseConfig
from .color_formatter import ColorFormatter
from .logger_config import LoggerConfig, LogLevel


def init_wandb(config: LoggerConfig, global_rank: Optional[int]) -> bool:
    os.environ["WANDB_BASE_URL"] = config.wandb_host
    os.environ["WANDB_API_KEY"] = config.wandb_api_key  # type: ignore
    group_name = config.wandb_group
    name = f"{socket.gethostname()}-{global_rank}" if group_name else None
    try:
        wandb.init(
            project=config.wandb_project,
            group=group_name,
            name=name,
            save_code=False,
            force=False,
            entity=config.wandb_team,
        )
        return True
    except wandb.UsageError:
        return False


class Logger:
    def __init__(
        self,
        config: LoggerConfig,
        name: Optional[str] = None,
        global_rank: Optional[int] = None,
    ) -> None:
        self._tensorboard_writer = None
        self._use_wandb = False
        self._logger = logging.getLogger(name="aleph-alpha-scaling")
        self._handler = logging.StreamHandler()
        self._file_handler = None
        self._logger.addHandler(self._handler)
        self.set_level(config.log_level)
        if config.log_dir is not None:
            # configure for distributed logging to files
            config.log_dir.mkdir(exist_ok=True, parents=True)
            self._file_handler = logging.FileHandler(filename=str(config.log_dir.absolute() / f"log_{name}.log"))
            self._logger.addHandler(self._file_handler)

            # configure tensorboard
            if config.use_tensorboard and config.is_rank_in_tensorboard_ranks(global_rank):
                self._tensorboard_writer = SummaryWriter(log_dir=str(config.log_dir / "tensorboard"))

        # configure wandb
        if config.use_wandb and config.is_rank_in_wandb_ranks(global_rank):
            self._use_wandb = init_wandb(config=config, global_rank=global_rank)

        # Write metrics for rank 0 if metrics ranks not set or if rank is included in metrics ranks
        self._write_metrics = config.is_rank_in_metrics_ranks(global_rank)
        self.set_formatter(name=name)

    def set_level(self, log_level: LogLevel) -> None:
        self._logger.setLevel(log_level.name)
        self._handler.setLevel(log_level.name)

    def set_formatter(self, name: Optional[str] = None) -> None:
        formatter = ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s")
        if name is not None:
            formatter = ColorFormatter(f"[%(asctime)s] [%(levelname)s] [{name}] %(message)s")

        self._handler.setFormatter(formatter)
        if self._file_handler is not None:
            self._file_handler.setFormatter(formatter)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if self._write_metrics:
            self.info(json.dumps(metrics))

        if self._use_wandb:
            wandb.log(metrics, step=step)

        if self._tensorboard_writer is not None:
            for k, v in metrics.items():
                self._tensorboard_writer.add_scalar(k, v, step)
            self._tensorboard_writer.flush()

    def log_config(self, config: BaseConfig) -> None:
        config_dict = copy.deepcopy(config.as_dict())
        self.log_config_dict(config_dict=config_dict)

    def log_config_dict(self, config_dict: dict) -> None:
        if self._use_wandb:
            wandb.config.update(config_dict, allow_val_change=True)

        if self._tensorboard_writer is not None:
            for name, value in config_dict.items():
                self._tensorboard_writer.add_text(name, str(value))

    def debug(self, msg: object) -> None:
        self._logger.debug(msg=msg)

    def info(self, msg: object) -> None:
        self._logger.info(msg=msg)

    def warning(self, msg: object) -> None:
        self._logger.warning(msg=msg)

    def error(self, msg: object) -> None:
        self._logger.error(msg=msg)

    def critical(self, msg: object) -> None:
        self._logger.critical(msg=msg)


class _LoggerSingleton:
    def __init__(self) -> None:
        self._logger_implementation: Optional[Logger] = Logger(LoggerConfig())

    def configure(
        self,
        config: LoggerConfig,
        name: Optional[str] = None,
        global_rank: Optional[int] = None,
    ) -> None:
        self._logger_implementation = Logger(config, name, global_rank)

    def __getattr__(self, name: str) -> Any:
        if self._logger_implementation is None:
            raise ValueError("Logger need to be initialised via configure(...)")
        # forward calls to the actual logger implementation
        return getattr(self._logger_implementation, name)


class _LoggerType(Logger, _LoggerSingleton):
    pass


logger: _LoggerType = _LoggerSingleton()  # type: ignore
