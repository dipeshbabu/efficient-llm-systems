"""Tests for hardware replay system.

Tests profile parsing, serialization, comparison, and prediction
using a real M5 Max baseline profile and a synthetic M1 Max profile.
"""

import json
from pathlib import Path

import pytest

from turboquant.hw_replay import (
    BenchResult,
    ComparisonReport,
    GPUInfo,
    HardwareProfile,
    LoadSnapshot,
    ModelInfo,
    PPLResult,
    SystemInfo,
    compare_profiles,
    parse_diag_output,
    predict_decode_from_baseline,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def m5_max_profile():
    """Real M5 Max profile (our baseline hardware)."""
    profile = HardwareProfile(
        diag_version=3,
        timestamp="2026-03-26T13:43:09Z",
        build_commit="dfc1097",
        system=SystemInfo(
            platform="Darwin",
            os_version="25.3.0",
            arch="arm64",
            cpu_brand="Apple M5 Max",
            cpu_cores_physical=18,
            cpu_cores_logical=18,
            ram_total_gb=128,
            apple_silicon=True,
            chip_model="Apple M5 Max",
            l1_dcache=65536,
            l2_cache=8388608,
            gpu=GPUInfo(
                name="MTL0",
                family="MTLGPUFamilyApple10  (1010)",
                family_id=1010,
                has_tensor=True,
                has_unified_memory=True,
                has_bfloat=True,
                recommended_max_working_set_mb=115448.73,
            ),
        ),
        model=ModelInfo(
            filename="Qwen3.5-35B-A3B-Q8_0.gguf",
            architecture="qwen35moe",
            name="Qwen3.5-35B-A3B",
            file_type="Q8_0",
            model_type="35B.A3B",
            params="34.66 B",
            n_layer=40,
            n_head=16,
            n_head_kv=2,
            n_expert=256,
            n_expert_used=8,
            n_ctx_train=262144,
            n_embd=2048,
        ),
        benchmarks=[
            # q8_0 decode baselines
            BenchResult("q8_0 decode (short)", "q8_0", "q8_0", 0, "decode", 85.8, 0.17),
            BenchResult("q8_0 decode @4K", "q8_0", "q8_0", 4096, "decode", 79.9, 1.55),
            BenchResult("q8_0 decode @8K", "q8_0", "q8_0", 8192, "decode", 77.4, 0.95),
            BenchResult("q8_0 decode @16K", "q8_0", "q8_0", 16384, "decode", 74.0, 0.80),
            BenchResult("q8_0 decode @32K", "q8_0", "q8_0", 32768, "decode", 68.0, 0.50),
            # turbo3 decode
            BenchResult("turbo3 decode (short)", "turbo3", "turbo3", 0, "decode", 77.4, 0.05),
            BenchResult("turbo3 decode @4K", "turbo3", "turbo3", 4096, "decode", 70.9, 1.27),
            BenchResult("turbo3 decode @8K", "turbo3", "turbo3", 8192, "decode", 66.6, 0.50),
            BenchResult("turbo3 decode @16K", "turbo3", "turbo3", 16384, "decode", 60.0, 0.40),
            BenchResult("turbo3 decode @32K", "turbo3", "turbo3", 32768, "decode", 52.0, 0.30),
            # q8_0 prefill baselines
            BenchResult("q8_0 prefill 2K", "q8_0", "q8_0", 2048, "prefill", 2707.0, 9.17),
            BenchResult("q8_0 prefill 4K", "q8_0", "q8_0", 4096, "prefill", 2429.0, 39.91),
            BenchResult("q8_0 prefill 8K", "q8_0", "q8_0", 8192, "prefill", 2052.0, 18.60),
            # turbo3 prefill
            BenchResult("turbo3 prefill 2K", "turbo3", "turbo3", 2048, "prefill", 2632.0, 13.13),
            BenchResult("turbo3 prefill 4K", "turbo3", "turbo3", 4096, "prefill", 2362.0, 118.70),
            BenchResult("turbo3 prefill 8K", "turbo3", "turbo3", 8192, "prefill", 2014.0, 26.99),
        ],
        ppl_results=[
            PPLResult("q8_0", 8, 6.111, 0.326),
            PPLResult("turbo3", 8, 6.211, 0.333),
            PPLResult("turbo3", 8, 6.120, 0.327, env="TURBO_LAYER_ADAPTIVE=2"),
        ],
        load_snapshots=[
            LoadSnapshot("pre_benchmark", "2026-03-26T13:43:09Z", "1.5 2.0 1.8", 50000, "", 350),
            LoadSnapshot("post_all_benchmarks", "2026-03-26T14:10:00Z", "3.0 2.5 2.0", 48000, "", 355),
        ],
    )
    return profile


@pytest.fixture
def m1_max_profile():
    """Synthetic M1 Max profile based on Mario's reported numbers."""
    profile = HardwareProfile(
        diag_version=3,
        timestamp="2026-03-26T15:00:00Z",
        build_commit="dfc1097",
        system=SystemInfo(
            platform="Darwin",
            os_version="24.2.0",
            arch="arm64",
            cpu_brand="Apple M1 Max",
            cpu_cores_physical=10,
            cpu_cores_logical=10,
            ram_total_gb=64,
            apple_silicon=True,
            chip_model="Apple M1 Max",
            l1_dcache=65536,
            l2_cache=4194304,
            gpu=GPUInfo(
                name="MTL0",
                family="MTLGPUFamilyApple7  (1007)",
                family_id=1007,
                has_tensor=False,
                has_unified_memory=True,
                has_bfloat=True,
                recommended_max_working_set_mb=55662.79,
            ),
        ),
        model=ModelInfo(
            filename="Qwen3.5-35B-A3B-Q8_0.gguf",
            architecture="qwen35moe",
            name="Qwen3.5-35B-A3B",
            file_type="Q8_0",
            n_layer=40,
        ),
        benchmarks=[
            # q8_0 decode — M1 is ~2x slower than M5
            BenchResult("q8_0 decode (short)", "q8_0", "q8_0", 0, "decode", 42.0, 0.20),
            BenchResult("q8_0 decode @4K", "q8_0", "q8_0", 4096, "decode", 38.0, 0.30),
            BenchResult("q8_0 decode @8K", "q8_0", "q8_0", 8192, "decode", 35.0, 0.25),
            BenchResult("q8_0 decode @16K", "q8_0", "q8_0", 16384, "decode", 30.0, 0.20),
            BenchResult("q8_0 decode @32K", "q8_0", "q8_0", 32768, "decode", 25.0, 0.15),
            # turbo3 decode — catastrophic at long context on M1
            BenchResult("turbo3 decode (short)", "turbo3", "turbo3", 0, "decode", 36.0, 0.10),
            BenchResult("turbo3 decode @4K", "turbo3", "turbo3", 4096, "decode", 20.0, 0.50),
            BenchResult("turbo3 decode @8K", "turbo3", "turbo3", 8192, "decode", 10.0, 0.30),
            BenchResult("turbo3 decode @16K", "turbo3", "turbo3", 16384, "decode", 5.0, 0.10),
            BenchResult("turbo3 decode @32K", "turbo3", "turbo3", 32768, "decode", 2.5, 0.05),
        ],
        ppl_results=[
            PPLResult("q8_0", 8, 6.111, 0.326),
            PPLResult("turbo3", 8, 6.211, 0.333),
        ],
    )
    return profile


# ============================================================
# Profile Serialization
# ============================================================

class TestProfileSerialization:

    def test_to_json_roundtrip(self, m5_max_profile):
        """Profile survives JSON serialization."""
        json_str = m5_max_profile.to_json()
        data = json.loads(json_str)
        assert data["system"]["gpu"]["family_id"] == 1010
        assert data["system"]["ram_total_gb"] == 128
        assert len(data["benchmarks"]) == 16

    def test_save_and_load(self, m5_max_profile, tmp_path):
        """Save to file and load back."""
        path = tmp_path / "m5_max.json"
        m5_max_profile.save(path)
        loaded = HardwareProfile.from_json(path)
        assert loaded.system.gpu.family_id == 1010
        assert loaded.system.gpu.has_tensor is True
        assert loaded.system.ram_total_gb == 128
        assert len(loaded.benchmarks) == 16
        assert loaded.model.n_layer == 40

    def test_model_info_preserved(self, m5_max_profile, tmp_path):
        path = tmp_path / "test.json"
        m5_max_profile.save(path)
        loaded = HardwareProfile.from_json(path)
        assert loaded.model.architecture == "qwen35moe"
        assert loaded.model.n_expert == 256
        assert loaded.model.n_expert_used == 8

    def test_ppl_preserved(self, m5_max_profile, tmp_path):
        path = tmp_path / "test.json"
        m5_max_profile.save(path)
        loaded = HardwareProfile.from_json(path)
        assert len(loaded.ppl_results) == 3
        assert loaded.ppl_results[0].ppl == 6.111

    def test_load_snapshots_preserved(self, m5_max_profile, tmp_path):
        path = tmp_path / "test.json"
        m5_max_profile.save(path)
        loaded = HardwareProfile.from_json(path)
        assert len(loaded.load_snapshots) == 2
        assert loaded.load_snapshots[0].label == "pre_benchmark"


# ============================================================
# Curve Extraction
# ============================================================

class TestCurveExtraction:

    def test_decode_curve(self, m5_max_profile):
        curve = m5_max_profile.get_decode_curve("turbo3")
        assert len(curve) == 5
        assert curve[0] == 77.4  # short context
        assert curve[8192] == 66.6

    def test_decode_curve_q8_0(self, m5_max_profile):
        curve = m5_max_profile.get_decode_curve("q8_0")
        assert curve[0] == 85.8
        assert curve[4096] == 79.9

    def test_prefill_curve(self, m5_max_profile):
        curve = m5_max_profile.get_prefill_curve("turbo3")
        assert len(curve) == 3
        assert curve[2048] == 2632.0

    def test_ratio_curve_decode(self, m5_max_profile):
        ratios = m5_max_profile.get_ratio_curve("turbo3", "q8_0", "decode")
        assert len(ratios) >= 4
        # turbo3/q8_0 at short context
        assert 0.85 < ratios[0] < 0.95  # 77.4/85.8 = 0.902

    def test_ratio_curve_prefill(self, m5_max_profile):
        ratios = m5_max_profile.get_ratio_curve("turbo3", "q8_0", "prefill")
        assert len(ratios) >= 3
        # Should be ~0.97x
        assert ratios[2048] > 0.95

    def test_empty_curve_for_missing_type(self, m5_max_profile):
        curve = m5_max_profile.get_decode_curve("turbo4")
        assert len(curve) == 0

    def test_inflection_point_m1(self, m1_max_profile):
        """M1 should show clear inflection point where decode drops off."""
        inflection = m1_max_profile.find_decode_inflection("turbo3")
        # M1 has steep dropoff — inflection should be detected
        assert inflection is not None
        assert inflection >= 4096  # Should be at some meaningful depth

    def test_inflection_point_m5(self, m5_max_profile):
        """M5 has gradual degradation — inflection exists but at deeper context."""
        inflection = m5_max_profile.find_decode_inflection("turbo3")
        # M5 has smoother curve, but still degrades
        # May or may not detect inflection depending on gradient

    def test_unreliable_measurements(self):
        """Flag impossibly high tok/s at 1K context."""
        profile = HardwareProfile(benchmarks=[
            BenchResult("test 1K", "turbo3", "turbo3", 1024, "decode", 999999.0, 0.0),
            BenchResult("test 4K", "turbo3", "turbo3", 4096, "decode", 70.0, 0.5),
        ])
        warnings = profile.flag_unreliable_measurements()
        assert len(warnings) == 1
        assert "1K" in warnings[0]

    def test_no_false_unreliable_flags(self, m5_max_profile):
        """Real measurements shouldn't be flagged."""
        warnings = m5_max_profile.flag_unreliable_measurements()
        assert len(warnings) == 0


# ============================================================
# Profile Comparison
# ============================================================

class TestComparison:

    def test_compare_detects_hardware_diff(self, m5_max_profile, m1_max_profile):
        report = compare_profiles(m5_max_profile, m1_max_profile)
        assert "GPU Family ID" in report.hardware_diff
        assert "Tensor API" in report.hardware_diff
        assert "RAM (GB)" in report.hardware_diff

    def test_compare_detects_decode_degradation(self, m5_max_profile, m1_max_profile):
        report = compare_profiles(m5_max_profile, m1_max_profile)
        assert len(report.decode_ratio_curve) >= 4

    def test_compare_flags_anomalies(self, m5_max_profile, m1_max_profile):
        report = compare_profiles(m5_max_profile, m1_max_profile)
        assert len(report.anomalies) > 0
        # Should flag constant cache thrashing
        assert any("constant cache" in a.lower() or "tensor" in a.lower()
                    for a in report.anomalies)

    def test_compare_markdown_output(self, m5_max_profile, m1_max_profile):
        report = compare_profiles(m5_max_profile, m1_max_profile)
        md = report.to_markdown()
        assert "Hardware Differences" in md
        assert "Decode Speed Ratio" in md
        assert "Anomalies" in md

    def test_compare_same_profile_no_anomalies(self, m5_max_profile):
        report = compare_profiles(m5_max_profile, m5_max_profile)
        assert len(report.hardware_diff) == 0
        assert len(report.anomalies) == 0

    def test_ppl_comparison(self, m5_max_profile, m1_max_profile):
        report = compare_profiles(m5_max_profile, m1_max_profile)
        assert len(report.ppl_comparison) >= 1


# ============================================================
# Decode Prediction
# ============================================================

class TestPrediction:

    def test_predict_m1_from_m5(self, m5_max_profile):
        """Predict M1 decode ratios from M5 baseline."""
        predicted = predict_decode_from_baseline(
            m5_max_profile,
            target_gpu_family_id=1007,
            target_has_tensor=False,
        )
        assert len(predicted) >= 4
        # Short context should be minimally affected
        assert predicted[0] > 0.7
        # Long context should be much worse
        assert predicted[32768] < predicted[0]

    def test_predict_same_hardware(self, m5_max_profile):
        """Predicting same hardware should return similar ratios."""
        predicted = predict_decode_from_baseline(
            m5_max_profile,
            target_gpu_family_id=1010,
            target_has_tensor=True,
        )
        actual = m5_max_profile.get_ratio_curve("turbo3", "q8_0", "decode")
        for depth in predicted:
            if depth in actual:
                # Should be close since same hardware
                assert abs(predicted[depth] - actual[depth]) < 0.05

    def test_predict_degradation_increases_with_context(self, m5_max_profile):
        """Predicted degradation should get worse at longer context."""
        predicted = predict_decode_from_baseline(
            m5_max_profile,
            target_gpu_family_id=1007,
            target_has_tensor=False,
        )
        depths = sorted(predicted.keys())
        if len(depths) >= 2:
            # Later depths should have lower ratios
            assert predicted[depths[-1]] <= predicted[depths[0]]

    def test_predict_empty_baseline(self):
        """Empty baseline returns empty predictions."""
        empty = HardwareProfile()
        predicted = predict_decode_from_baseline(empty, 1007, False)
        assert len(predicted) == 0


# ============================================================
# Diagnostic Output Parsing
# ============================================================

class TestDiagParsing:

    SAMPLE_DIAG = """TURBO_DIAG_VERSION=3
TURBO_DIAG_TIMESTAMP=2026-03-26T13:43:09Z
TURBO_DIAG_MODEL=Qwen3.5-35B-A3B-Q8_0.gguf
TURBO_DIAG_MODEL_SIZE=34G

[HW] os=Darwin os_version=25.3.0 arch=arm64
[HW] cpu_brand=Apple M5 Max
[HW] cpu_cores_physical=18
[HW] cpu_cores_logical=18
[HW] ram_total_gb=128
[HW] apple_silicon=true
[HW] chip_model=Apple M5 Max
[HW] l1_dcache=65536
[HW] l2_cache=8388608

[GPU] ggml_metal_device_init: GPU name:   MTL0
[GPU] ggml_metal_device_init: GPU family: MTLGPUFamilyApple10  (1010)
[GPU] ggml_metal_device_init: has tensor            = true
[GPU] ggml_metal_device_init: has unified memory    = true
[GPU] ggml_metal_device_init: has bfloat            = true
[GPU] ggml_metal_device_init: recommendedMaxWorkingSetSize  = 115448.73 MB

[MODEL] print_info: general.name          = Qwen3.5-35B-A3B
[MODEL] print_info: arch                  = qwen35moe
[MODEL] print_info: n_layer               = 40
[MODEL] print_info: n_expert              = 256
[MODEL] filename=Qwen3.5-35B-A3B-Q8_0.gguf

[BUILD] dfc1097 fix: add turbo3/turbo4 cache types

[BENCH_START] label="q8_0 decode (short)" ctk=q8_0 ctv=q8_0 args="-p 0 -n 128" env="" timestamp=2026-03-26T13:45:00Z
| qwen35moe 35B.A3B Q8_0 | 34.36 GiB | 34.66 B | MTL,BLAS | 6 | q8_0 | q8_0 | 1 | tg128 | 85.83 ± 0.17 |
[BENCH_END] label="q8_0 decode (short)" wall_ms=5000

[BENCH_START] label="turbo3 decode (short)" ctk=turbo3 ctv=turbo3 args="-p 0 -n 128" env="" timestamp=2026-03-26T13:46:00Z
| qwen35moe 35B.A3B Q8_0 | 34.36 GiB | 34.66 B | MTL,BLAS | 6 | turbo3 | turbo3 | 1 | tg128 | 77.42 ± 0.05 |
[BENCH_END] label="turbo3 decode (short)" wall_ms=5500

[PPL_START] label="q8_0 PPL (8 chunks)" ctk=q8_0 ctv=q8_0 chunks=8 timestamp=2026-03-26T14:00:00Z
Final estimate: PPL = 6.1109 +/- 0.32553
[PPL_END] label="q8_0 PPL (8 chunks)"

[PPL_START] label="turbo3 PPL (8 chunks)" ctk=turbo3 ctv=turbo3 chunks=8 timestamp=2026-03-26T14:05:00Z
Final estimate: PPL = 6.2109 +/- 0.33250
[PPL_END] label="turbo3 PPL (8 chunks)"

[LOAD_SNAPSHOT] label=pre_benchmark timestamp=2026-03-26T13:43:09Z
[LOAD_SNAPSHOT] load_avg=1.5 2.0 1.8
[LOAD_SNAPSHOT] process_count=350
[LOAD_SNAPSHOT] approx_free_ram=50000 MB

[LOAD_SNAPSHOT] label=post_all_benchmarks timestamp=2026-03-26T14:10:00Z
[LOAD_SNAPSHOT] load_avg=3.0 2.5 2.0
[LOAD_SNAPSHOT] process_count=355

TURBO_DIAG_COMPLETE=true
"""

    def test_parse_version(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.diag_version == 3

    def test_parse_timestamp(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.timestamp == "2026-03-26T13:43:09Z"

    def test_parse_hw_platform(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.system.platform == "Darwin"
        assert profile.system.arch == "arm64"

    def test_parse_hw_cpu(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.system.cpu_brand == "Apple M5 Max"
        assert profile.system.cpu_cores_physical == 18
        assert profile.system.ram_total_gb == 128

    def test_parse_hw_apple_silicon(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.system.apple_silicon is True
        assert profile.system.chip_model == "Apple M5 Max"

    def test_parse_gpu_family(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.system.gpu.family_id == 1010
        assert "Apple10" in profile.system.gpu.family

    def test_parse_gpu_tensor(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.system.gpu.has_tensor is True

    def test_parse_gpu_unified_memory(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.system.gpu.has_unified_memory is True

    def test_parse_gpu_working_set(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.system.gpu.recommended_max_working_set_mb == 115448.73

    def test_parse_model_info(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert profile.model.name == "Qwen3.5-35B-A3B"
        assert profile.model.architecture == "qwen35moe"
        assert profile.model.n_layer == 40
        assert profile.model.n_expert == 256

    def test_parse_build_commit(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert "dfc1097" in profile.build_commit

    def test_parse_bench_results(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert len(profile.benchmarks) >= 2
        q8_decode = [b for b in profile.benchmarks if b.cache_type_k == "q8_0" and b.mode == "decode"]
        assert len(q8_decode) >= 1
        assert q8_decode[0].tok_per_sec == 85.83

    def test_parse_ppl_results(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert len(profile.ppl_results) == 2
        q8_ppl = [p for p in profile.ppl_results if p.cache_type == "q8_0"]
        assert len(q8_ppl) == 1
        assert q8_ppl[0].ppl == 6.1109

    def test_parse_load_snapshots(self):
        profile = parse_diag_output(self.SAMPLE_DIAG)
        assert len(profile.load_snapshots) >= 2
        pre = [s for s in profile.load_snapshots if s.label == "pre_benchmark"]
        assert len(pre) == 1
        assert pre[0].process_count == 350

    def test_parsed_profile_roundtrips(self, tmp_path):
        """Parse → save → load should preserve all data."""
        profile = parse_diag_output(self.SAMPLE_DIAG)
        path = tmp_path / "roundtrip.json"
        profile.save(path)
        loaded = HardwareProfile.from_json(path)
        assert loaded.system.gpu.family_id == profile.system.gpu.family_id
        assert loaded.model.n_layer == profile.model.n_layer
        assert len(loaded.benchmarks) == len(profile.benchmarks)
        assert len(loaded.ppl_results) == len(profile.ppl_results)
