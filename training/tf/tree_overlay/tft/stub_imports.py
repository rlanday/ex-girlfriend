"""import stub for GPU-only libraries (Python 3.13, no torch/fla/triton needed)

`from tft import stub_imports` BEFORE the first `import torch`, and every
module of the prefix list below resolves to a MagicMock package.  This lets the
dry runs check syntax, imports and pure-Python logic on a machine without CUDA;
all of them use this one stub.

Limits: a MagicMock cannot serve as a class base, so `import pysrc.model`
still fails at `class CastedLinear(Module)`.
Everything that does not subclass a torch class works.
"""

import importlib.util
import sys
from unittest.mock import MagicMock


class StubLoader:
    def create_module(self, spec):
        m = MagicMock()
        m.__name__ = spec.name
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


class StubFinder:
    prefixes = (
        "fla",
        "torch",
        "tqdm",
        "pytorch_optimizer",
        "einops",
        "triton",
        "jaxtyping",
        "wandb",
        "datasets",
        "transformers",
        "safetensors",
    )

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.prefixes:
            return importlib.util.spec_from_loader(name, StubLoader(), is_package=True)
        return None


def install() -> None:
    if not any(isinstance(f, StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, StubFinder())


install()
