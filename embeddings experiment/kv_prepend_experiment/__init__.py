"""KV-prepend probing experiment package.

The implementation is intentionally self-contained inside the
`embeddings experiment` folder so it can be copied to a remote machine
without relying on the rest of this repository.
"""

from .config import ExperimentSpec, load_experiment_spec

__all__ = ["ExperimentSpec", "load_experiment_spec"]
