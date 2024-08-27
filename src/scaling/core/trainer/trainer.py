import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import torch

from scaling.core.data import BaseDatasetBatchGeneric
from scaling.core.logging import logger
from scaling.core.nn import ParallelSelfAttention
from scaling.core.nn.parallel_module import (
    BaseLossInputGeneric,
    EvaluationStepOutput,
    TrainStepOutput,
)
from scaling.core.topology import Topology

from ..context import BaseContext
from ..data import (
    BaseDataset,
    DataLoader,
)
from ..nn import ParallelModule
from ..optimizer import BaseOptimizer
from .trainer_config import TrainerConfig

BaseContextGeneric = TypeVar("BaseContextGeneric", bound=BaseContext)
ParallelModuleGeneric = TypeVar("ParallelModuleGeneric", bound=ParallelModule)


class BaseTrainer(Generic[BaseContextGeneric, ParallelModuleGeneric]):
    def __init__(
        self,
        config: TrainerConfig,
        context: BaseContextGeneric,
        parallel_module: ParallelModuleGeneric,
        optimizer: BaseOptimizer,
        dataset: Optional[BaseDataset],
        sync_batch_to_model_parallel: Callable[[Topology, Optional[BaseDatasetBatchGeneric]], BaseDatasetBatchGeneric],
        loss_function: Callable[
            [
                BaseLossInputGeneric,
                Any,
            ],  # TODO Any -> Optional[BaseDatasetBatchGeneric] or something
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        ],
        metrics_aggregation_fn: Optional[Callable] = None,
        dataset_evaluation: Optional[BaseDataset] = None,
    ):
        self.config = config
        self.context = context
        self.parallel_module = parallel_module

        (
            self.parameters_total,
            self.parameters_unique,
        ) = self.parallel_module.get_params_count()
        logger.log_config_dict(
            {
                "parameters_total": self.parameters_total,
                "parameters_unique": self.parameters_unique,
            }
        )

        self.optimizer = optimizer
        self.dataset = dataset
        self.dataset_evaluation = dataset_evaluation

        # potentially load checkpoint
        checkpoint_loaded: bool = self.load_checkpoint(
            load_dir=self.config.load_dir,
            load_optimizer_states=self.config.load_optimizer_states,
            load_context=self.config.load_context,
            allowed_missing_keys_in_checkpoint=self.config.allowed_missing_keys_in_checkpoint,
            allowed_unexpected_keys_in_checkpoint=self.config.allowed_unexpected_keys_in_checkpoint,
            ignore_keys_in_checkpoint=self.config.ignore_keys_in_checkpoint,
        )
        if config.assert_checkpoint_loaded:
            assert checkpoint_loaded, (
                "checkpoint could not be loaded. "
                "if this is intended "
                "you may change the parameter 'assert_checkpoint_loaded' in the TrainerConfig to False."
            )

        if config.merge_lora_after_loading_checkpoint:
            for module in parallel_module.modules():
                if isinstance(module, ParallelSelfAttention):
                    module.merge_lora_weights()
            self.optimizer.refresh_optimizer_after_model_change()
            logger.info("Merged LoRa weights")

        # initialize data loader class after loading of checkpoint
        # to make sure that consumed samples are loaded from checkpoint
        if self.context.topology.is_io_rank:
            assert self.dataset is not None
            self.dataloader: Optional[DataLoader] = DataLoader(
                seed=self.config.seed,
                consumed_samples=self.context.consumed_samples,
                dataset=self.dataset,
                topology=self.context.topology,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                prefetch_factor=self.config.dataloader_prefetch_factor,
            )
            if dataset_evaluation is not None:
                assert self.dataset_evaluation is not None
                self.dataloader_evaluation: Optional[DataLoader] = DataLoader(
                    seed=self.config.seed,
                    consumed_samples=self.context.consumed_samples_evaluation,
                    dataset=self.dataset_evaluation,
                    topology=self.context.topology,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=self.config.dataloader_pin_memory,
                    prefetch_factor=self.config.dataloader_prefetch_factor,
                )
            else:
                self.dataloader_evaluation = None
        else:
            self.dataloader = None
            self.dataloader_evaluation = None

        self.sync_batch_to_model_parallel = sync_batch_to_model_parallel
        self.loss_function = loss_function
        self.metrics_aggregation_fn = metrics_aggregation_fn

        if self.config.delete_past_optimizer_states:
            with open(Path(__file__).parent / "warnings.txt") as in_f:
                warnings = in_f.read().split("===")

            logger.warning("\n" + random.choice(warnings))
            logger.warning(
                "DELETE_PAST_OPTIMIZER_STATES IS SET TO TRUE, "
                "THIS WILL DELETE OPTIMIZER STATES FROM ALL CHECKPOINTS EXCEPT THE LATEST ONE. "
                "THIS HAPPENS EVERY TIME A NEW CHECKPOINT IS SAVED. "
                "IF YOU WANT TO KEEP ALL OPTIMIZER STATES, SET THIS FLAG TO FALSE."
            )
            pass

    def save_checkpoint(self, save_dir: Optional[Path] = None) -> Path:
        save_dir = save_dir or self.config.save_dir
        assert save_dir is not None
        iteration_dir: Path = save_dir / f"global_step{self.context.iterations}"
        iteration_dir.mkdir(exist_ok=True, parents=True)
        self.parallel_module.save_checkpoint(
            iteration_dir,
            separate_file_for_parameters=self.config.separate_file_for_parameters,
        )
        self.optimizer.save_checkpoint(iteration_dir)
        self.context.save_checkpoint(iteration_dir)

        if self.context.topology.config.global_rank == 0:
            with open(save_dir / "latest", "w", encoding="UTF-8") as f:
                f.write(f"global_step{self.context.iterations}")

        logger.info(f"saved checkpoint: {iteration_dir}")

        return save_dir

    def load_checkpoint(
        self,
        load_dir: Optional[Path] = None,
        load_optimizer_states: bool = True,
        load_context: bool = True,
        allowed_missing_keys_in_checkpoint: Optional[List[str]] = None,
        allowed_unexpected_keys_in_checkpoint: Optional[List[str]] = None,
        ignore_keys_in_checkpoint: Optional[List[str]] = None,
    ) -> bool:
        if load_dir is None:
            return False

        if (load_dir / "latest").is_file():
            with open(load_dir / "latest", "r", encoding="UTF-8") as f:
                global_step_dir = (
                    f.read().strip()  # strip removes potential line breaks and spaces
                )

            iteration_dir = load_dir / global_step_dir
        elif len(list((load_dir.glob("*.pt")))) > 0:
            logger.info(f"no latest file found, using load dir directly instead: {load_dir}")
            iteration_dir = load_dir
        else:
            logger.error(f"no files found in load dir: {load_dir}")
            return False

        if not iteration_dir.is_dir():
            logger.error(f"iteration_dir does not exist: {iteration_dir}")
            return False

        self.parallel_module.load_checkpoint(
            iteration_dir,
            allowed_missing_keys_in_checkpoint=allowed_missing_keys_in_checkpoint,
            allowed_unexpected_keys_in_checkpoint=allowed_unexpected_keys_in_checkpoint,
            ignore_keys_in_checkpoint=ignore_keys_in_checkpoint,
        )

        if load_optimizer_states:
            self.optimizer.load_checkpoint(iteration_dir)
        else:
            # Refresh copied tensors after model has been updated
            self.optimizer.refresh_optimizer_after_model_change()
        if load_context:
            self.context.load_checkpoint(iteration_dir)

        logger.info(f"loaded checkpoint: {iteration_dir}")
        return True

    def train_step(self) -> TrainStepOutput:
        train_step_output = self.parallel_module.train_step(
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            sync_batch_to_model_parallel=self.sync_batch_to_model_parallel,
            loss_function=self.loss_function,
            metrics_aggregation_fn=self.metrics_aggregation_fn,
        )

        self.context.step()

        return train_step_output

    def eval_step(self) -> EvaluationStepOutput:
        if self.context.topology.is_io_rank:
            assert self.dataloader_evaluation is not None, (
                "needs an evaluation dataloader on io ranks if evaluation is to be performed. "
                "Remember to give the trainer a dataset_evaluation"
            )
        eval_step_output = self.parallel_module.evaluation_step(
            dataloader=self.dataloader_evaluation,
            sync_batch_to_model_parallel=self.sync_batch_to_model_parallel,
            loss_function=self.loss_function,
            metrics_aggregation_fn=self.metrics_aggregation_fn,
        )

        return eval_step_output

    def log_metrics(
        self,
        train_step_output: TrainStepOutput,
        eval_step_output: Optional[EvaluationStepOutput],
    ) -> dict[str, Any]:
        logger.info(f"completed step {self.context.iterations}")

        metrics: dict[str, Optional[int | float]] = dict()
        if train_step_output.metrics:
            for k, v in train_step_output.metrics.items():
                metrics[f"training/{k}"] = v
        assert "training/loss" not in metrics
        metrics["training/loss"] = train_step_output.loss
        metrics["training/step_duration"] = train_step_output.step_duration

        if train_step_output.global_grad_norm is not None:
            metrics["training/global_grad_norm"] = train_step_output.global_grad_norm
        if train_step_output.global_grad_norm_clipped is not None:
            metrics["training/global_grad_norm_clipped"] = train_step_output.global_grad_norm_clipped
        if train_step_output.learning_rates is not None:
            for (
                param_group_name,
                learning_rate,
            ) in train_step_output.learning_rates.items():
                metrics[f"training/learning_rate_{param_group_name}"] = learning_rate
        if train_step_output.overflow is not None:
            metrics["training/overflow"] = int(train_step_output.overflow)
        if train_step_output.no_overflow_steps is not None:
            metrics["training/no_overflow_steps"] = train_step_output.no_overflow_steps
        if train_step_output.current_loss_scale is not None:
            metrics["training/current_loss_scale"] = train_step_output.current_loss_scale

        if eval_step_output is not None:
            if eval_step_output.metrics:
                for k, v in eval_step_output.metrics.items():
                    metrics[f"evaluation/{k}"] = v

            metrics["evaluation/loss"] = eval_step_output.loss
            metrics["evaluation/step_duration"] = eval_step_output.step_duration

        logger.log_metrics(metrics, step=self.context.iterations)

        return metrics

    def run_training(self, return_metrics: bool = False) -> Optional[List[Dict[str, Union[float, int]]]]:
        metrics_list: List[Dict[str, Any]] = list()
        while self.context.iterations < (self.config.train_iterations or 0):
            # model train step
            train_step_output = self.train_step()
            # save checkpoint
            if (
                self.config.save_interval is not None
                and self.config.save_dir is not None
                and self.context.iterations % self.config.save_interval == 0
            ):
                self.save_checkpoint()
            # model eval step
            if self.config.eval_interval is not None and self.context.iterations % self.config.eval_interval == 0:
                eval_step_output = self.eval_step()
            else:
                eval_step_output = None
            # log metrics
            if self.context.topology.config.global_rank == 0:
                metrics = self.log_metrics(
                    train_step_output=train_step_output,
                    eval_step_output=eval_step_output,
                )

                if return_metrics:
                    metrics_list.append(metrics)

        if return_metrics:
            return metrics_list
        else:
            return None
