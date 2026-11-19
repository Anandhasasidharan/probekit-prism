"""Apply the same regularized probe learner to pooled SAE features."""

import numpy as np

from probekit.probes.datasets import ProbeConfig, ProbeDataset
from probekit.probes.raw_trainer import LinearProbe, train_raw_probe


def train_sae_probe(
    data: ProbeDataset,
    train: np.ndarray,
    val: np.ndarray,
    config: ProbeConfig,
    seed: int,
) -> LinearProbe:
    targets = data.labels
    return train_raw_probe(
        data.sae[train], targets[train], data.sae[val], targets[val], config, seed
    )
