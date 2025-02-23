# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from NVIDIA CORPORATION
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Optional

import torch

from ...topology import Topology
from ..parameter_meta import (
    CoreParameterMeta,
)
from .utils import all_reduce, all_reduce_scatter_to_sequence_parallel


class VocabParallelEmbedding(torch.nn.Module):
    """
    Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        finetunable_token_ids: list[int],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        topology: Optional[Topology] = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
    ) -> None:
        """
        num_embeddings (`int`)
            vocabulary size
        embedding_dim (`int`)
            size of hidden state
        device (`Optional[torch.device]`)
            Current torch device
        dtype (`torch.dtype`)
            Data type of created weights and biases of linear layer
            Default: torch.float32
        topology (`Optional[Topology]`)
            Layout on nodes
        init_method (`Callable[[torch.Tensor], torch.Tensor]`)
            initialization method for embedding layers
            Default: torch.nn.init.xavier_normal_
        """
        super().__init__()

        # remember parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        assert not (topology is not None and device is not None), "cannot specify both device and topology"
        if topology is not None:
            self._device = topology.device
        elif device is not None:
            self._device = device
        else:
            self._device = torch.device("cuda", torch.cuda.current_device())

        self.dtype = dtype
        self.topology = topology
        self.init_method = init_method

        # determine size for active model parallel
        self.model_parallel_size = 1 if self.topology is None else self.topology.config.model_parallel_size

        # determine number of embeddings per partition
        assert self.num_embeddings % self.model_parallel_size == 0, (
            f"cannot parallelize embedding, num_embeddings ({self.num_embeddings}) "
            f"needs to be divisible by model parallel size ({self.model_parallel_size})"
        )
        self.vocab_size_per_partition = self.num_embeddings // self.model_parallel_size
        self.vocab_start_index = (
            0 if topology is None else (topology.model_parallel_rank * self.vocab_size_per_partition)
        )
        self.vocab_end_index = self.vocab_start_index + self.vocab_size_per_partition

        self.weight = torch.nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition,
                self.embedding_dim,
                device=self._device,
                dtype=self.dtype,
            )
        )
        # initialize weights
        init_method(self.weight)
        CoreParameterMeta.register_on_parameter(
            parameter=self.weight,
            is_model_parallel=True,
            model_parallel_dimension=0,
        )

        if len(finetunable_token_ids) > 0:
            mask = torch.zeros_like(self.weight)
            for token_id in finetunable_token_ids:
                if topology is not None and (
                    (self.vocab_size_per_partition * topology.model_parallel_rank)
                    <= token_id
                    < (self.vocab_size_per_partition * (topology.model_parallel_rank + 1))
                ):
                    mask[token_id - (self.vocab_size_per_partition * topology.model_parallel_rank), :] = 1

            def weight_hook(grad: torch.Tensor | None) -> torch.Tensor | None:
                if grad is None:
                    return None
                else:
                    return mask * grad

            self.weight.register_hook(weight_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        if self.model_parallel_size > 1:
            assert self.topology is not None
            # Build the mask.
            input_mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)
            # Mask the input.
            masked_input = x - self.vocab_start_index
            masked_input[input_mask] = 0
            output_parallel = torch.nn.functional.embedding(
                masked_input,
                weight,
            )
            # Mask the output embedding.
            if self.model_parallel_size > 1:
                output_parallel[input_mask, :] = 0.0
            # Reduce across all the model parallel GPUs and optionally scatter across sequence dimension.
            if self.topology.config.sequence_parallel:
                output = all_reduce_scatter_to_sequence_parallel(output_parallel, topology=self.topology)
            else:
                output = all_reduce(output_parallel, topology=self.topology)
        else:
            output = torch.nn.functional.embedding(
                x,
                weight,
            )

        return output
