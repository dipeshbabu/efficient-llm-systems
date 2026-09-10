"""Exercise benchmark loading without the optional Accelerate dispatcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(path):
    spec = importlib.util.spec_from_file_location("benchmark_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def single_device_stack(monkeypatch):
    # Initialize optional Torch support before substituting the benchmark's
    # loading boundary so the stand-in cannot leak into other component tests.
    importlib.import_module("turboquant")
    importlib.import_module("turboquant.outlier")

    class Model:
        device = None
        training = True

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            assert self.device is not None, "model was not placed before evaluation"
            self.training = False
            return self

        def parameters(self):
            return [SimpleNamespace(numel=lambda: 10)]

    models = []

    def from_pretrained(*args, **kwargs):
        if kwargs.get("device_map") is not None:
            raise ImportError("device_map requires Accelerate")
        model = Model()
        models.append(model)
        return model

    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3])
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(float16="float16", float32="float32")
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        ),
    )
    return models, tokenizer


def test_real_model_validator_loads_on_cpu_without_accelerate(single_device_stack):
    models, tokenizer = single_device_stack
    module = _module(
        ROOT
        / "components/turboquant-reference/benchmarks/validation/validate_real_model.py"
    )
    loaded, returned_tokenizer = module.load_model()
    assert loaded is models[0]
    assert loaded.device == "cpu"
    assert loaded.training is False
    assert returned_tokenizer is tokenizer


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_skip_rate_loader_preserves_selected_device(single_device_stack, device):
    models, _ = single_device_stack
    module = _module(ROOT / "tools/validation/measure_skip_rate.py")
    assert module.measure_skip_rates(context_lengths=[], device=device) == []
    assert models[0].device == device
    assert models[0].training is False
