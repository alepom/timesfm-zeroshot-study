"""tfmstudy: code for the TimesFM 3 zero-shot forecasting study."""

import os

# Expandable CUDA segments reduce fragmentation on the 15 GB T4 (large multivariate fev tasks ran out of memory in
# version 0.5.1). Must be set before torch is imported; it does not change any forecast.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.5.5"

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_INDEX = 4
