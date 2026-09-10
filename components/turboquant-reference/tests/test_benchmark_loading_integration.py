"""Qualify real single-device benchmark loading with no Accelerate installed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(path):
    spec = importlib.util.spec_from_file_location("benchmark_integration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_sharded_checkpoint_loads_and_runs_without_accelerate(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    if importlib.util.find_spec("accelerate") is not None:
        pytest.skip("use an isolated benchmark environment without Accelerate")

    # Generate a tiny local checkpoint: no Hub credentials or model downloads.
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=8,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.GPT2LMHeadModel(config)
    model.save_pretrained(tmp_path, max_shard_size="1KB", safe_serialization=True)
    assert (tmp_path / "model.safetensors.index.json").is_file()
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "[PAD]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[UNK]": 3,
                "The": 4,
                "quick": 5,
                "brown": 6,
                "fox": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    ).save_pretrained(tmp_path)

    validator = _module(
        ROOT
        / "components/turboquant-reference/benchmarks/validation/validate_real_model.py"
    )
    validator.MODEL_NAME = str(tmp_path)
    loaded, loaded_tokenizer = validator.load_model()
    assert next(loaded.parameters()).device.type == "cpu"
    assert next(loaded.parameters()).dtype == torch.float32
    assert not loaded.training
    kv = validator.extract_kv_cache(loaded, loaded_tokenizer, "The quick brown fox")
    assert kv["k_cache"].shape == (2, 2, 4, 8)
    assert kv["v_cache"].shape == (2, 2, 4, 8)
    assert np.isfinite(kv["k_cache"]).all()
    assert np.isfinite(kv["v_cache"]).all()

    measurement = _module(ROOT / "tools/validation/measure_skip_rate.py")
    results = measurement.measure_skip_rates(
        model_name=str(tmp_path), context_lengths=[4], device="cpu"
    )
    assert len(results) == 1
    assert results[0]["context_length"] == 4
    assert results[0]["total_positions"] == 16
    assert 0 <= results[0]["overall_skip_rate"] <= 1
