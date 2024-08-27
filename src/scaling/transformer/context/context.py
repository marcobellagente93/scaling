from scaling.core import Topology

from .config import TransformerConfig


class TransformerContext:
    config: TransformerConfig

    def __init__(
        self,
        config: TransformerConfig,
        topology: Topology,
    ) -> None:
        super().__init__(config=config, topology=topology)
