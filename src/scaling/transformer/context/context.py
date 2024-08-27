from scaling.core import BaseContext, Topology

from .config import TransformerConfig


class TransformerContext(BaseContext):
    config: TransformerConfig

    def __init__(
        self,
        config: TransformerConfig,
        topology: Topology,
    ) -> None:
        super().__init__(config=config, topology=topology)
