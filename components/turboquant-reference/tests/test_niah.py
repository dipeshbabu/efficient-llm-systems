"""Tests for the NIAH validation tool.

Kamradt / RULER methodology. Mocks ALL subprocess and HTTP calls so no
llama-server is needed.
"""

from __future__ import annotations

import importlib.util
import json
import random
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import niah_test.py as a module
# ---------------------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "niah",
    str(Path(__file__).resolve().parents[3] / "tools" / "validation" / "niah_test.py"),
)
niah = importlib.util.module_from_spec(spec)
# Register in sys.modules so dataclasses can resolve the module's __dict__
sys.modules["niah"] = niah

# Prevent atexit/signal registration from firing during import
with patch("atexit.register"), patch("signal.signal"):
    spec.loader.exec_module(niah)


# ===================================================================
# Filler text generation
# ===================================================================


class TestBuildFiller:
    """Tests for _build_filler()."""

    def test_reaches_target_size(self):
        """Filler paragraphs should reach the target character count."""
        rng = random.Random(42)
        paragraphs = niah._build_filler(4000, rng)
        total = sum(len(p) for p in paragraphs) + 2 * (len(paragraphs) - 1)
        assert total >= 4000

    def test_diverse_topics(self):
        """Paragraphs should be shuffled — first paragraph shouldn't always be the same."""
        rng1 = random.Random(1)
        rng2 = random.Random(99)
        p1 = niah._build_filler(2000, rng1)
        p2 = niah._build_filler(2000, rng2)
        # Different seeds should shuffle differently
        assert p1[0] != p2[0]

    def test_not_repetitive_within_24(self):
        """First 24 paragraphs should all be unique (no repeats in one cycle)."""
        rng = random.Random(42)
        paragraphs = niah._build_filler(50000, rng)
        # The first 24 should all be unique (24 = len(FILLER_PARAGRAPHS))
        first_cycle = paragraphs[:24]
        assert len(set(first_cycle)) == 24

    def test_small_target(self):
        """Even a small target should produce at least one paragraph."""
        rng = random.Random(42)
        paragraphs = niah._build_filler(10, rng)
        assert len(paragraphs) >= 1


# ===================================================================
# Magic number generation
# ===================================================================


class TestMakeMagicNumber:
    """Tests for _make_magic_number()."""

    def test_seven_digits(self):
        rng = random.Random(42)
        num = niah._make_magic_number(rng)
        assert len(num) == 7
        assert num.isdigit()

    def test_range(self):
        rng = random.Random(42)
        for _ in range(100):
            num = niah._make_magic_number(rng)
            assert 1000000 <= int(num) <= 9999999

    def test_deterministic(self):
        assert niah._make_magic_number(random.Random(42)) == niah._make_magic_number(
            random.Random(42)
        )


# ===================================================================
# Needle data class
# ===================================================================


class TestNeedle:
    """Tests for the Needle dataclass."""

    def test_sentence_format(self):
        n = niah.Needle(
            key="The special magic number is", value="1234567", depth_pct=0.5
        )
        assert n.sentence == "The special magic number is: 1234567."

    def test_custom_key(self):
        n = niah.Needle(key="The secret password is", value="9999999", depth_pct=0.0)
        assert n.sentence == "The secret password is: 9999999."


# ===================================================================
# Needle generation & depth positions
# ===================================================================


class TestNeedleGeneration:
    """Tests for needle construction patterns used across modes."""

    def test_single_mode_needle_depth(self):
        """Single mode creates a needle at depth_pct / 100."""
        depth_pct = 50
        needle = niah.Needle(
            key="The special magic number is",
            value=niah._make_magic_number(random.Random(42)),
            depth_pct=depth_pct / 100.0,
        )
        assert needle.depth_pct == 0.5

    def test_multi_key_distractor_positions(self):
        """Distractors should be spread across 0-1 range."""
        num_distractors = 5
        positions = [(i + 1) / (num_distractors + 2) for i in range(num_distractors)]
        # All positions should be between 0 and 1 exclusive
        for p in positions:
            assert 0.0 < p < 1.0
        # Positions should be monotonically increasing
        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1]

    def test_multi_value_positions(self):
        """Multi-value needles should be evenly spread."""
        vc = 4
        positions = [(i + 1) / (vc + 1) for i in range(vc)]
        assert len(positions) == 4
        for p in positions:
            assert 0.0 < p < 1.0


# ===================================================================
# Distractor generation (multi-key mode)
# ===================================================================


class TestDistractors:
    """Tests for distractor needle generation in multi-key mode."""

    def test_distractor_keys_from_list(self):
        """Distractors should use DISTRACTOR_KEYS."""
        rng = random.Random(42)
        num_distractors = 3
        distractors = []
        for i in range(num_distractors):
            d_key = niah.DISTRACTOR_KEYS[i % len(niah.DISTRACTOR_KEYS)]
            distractors.append(
                niah.Needle(
                    key=d_key,
                    value=niah._make_magic_number(rng),
                    depth_pct=(i + 1) / (num_distractors + 2),
                )
            )
        assert len(distractors) == 3
        assert distractors[0].key == "The secret password is"
        assert distractors[1].key == "The hidden code is"
        assert distractors[2].key == "The encrypted token is"

    def test_distractor_values_are_7_digit(self):
        rng = random.Random(42)
        for _i in range(5):
            val = niah._make_magic_number(rng)
            assert len(val) == 7
            assert val.isdigit()

    def test_distractor_key_wraps(self):
        """With more distractors than keys, should wrap around."""
        num_keys = len(niah.DISTRACTOR_KEYS)
        key = niah.DISTRACTOR_KEYS[(num_keys + 2) % num_keys]
        assert key == niah.DISTRACTOR_KEYS[2]


# ===================================================================
# Scoring — 7-digit exact match
# ===================================================================


class TestScoreSingle:
    """Tests for _score_single()."""

    def test_exact_match(self):
        assert niah._score_single("1234567", "1234567") is True

    def test_code_in_sentence(self):
        assert niah._score_single("The number is 1234567, I think.", "1234567") is True

    def test_no_match(self):
        assert niah._score_single("I don't know", "1234567") is False

    def test_reject_6_digit(self):
        """6-digit number should NOT match a 7-digit expected value."""
        assert niah._score_single("123456", "1234567") is False

    def test_reject_partial(self):
        """Partial overlap should not match."""
        assert niah._score_single("1234568", "1234567") is False

    def test_reject_8_digit_containing_7(self):
        """An 8-digit number that contains the 7-digit code shouldn't match (word boundary)."""
        assert niah._score_single("12345678", "1234567") is False

    def test_empty_response(self):
        assert niah._score_single("", "1234567") is False

    def test_garbage_response(self):
        assert niah._score_single("asdfghjkl!@#$%^", "9999999") is False

    def test_number_with_extra_text(self):
        assert niah._score_single("Sure! The answer is 5551234.", "5551234") is True

    def test_repeated_wrong_numbers(self):
        assert niah._score_single("1111111 2222222 3333333", "4444444") is False


# ===================================================================
# Multi-value scoring
# ===================================================================


class TestScoreMultiValue:
    """Tests for _score_multi_value()."""

    def test_all_found(self):
        result = niah._score_multi_value(
            "1234567, 7654321, 1111111", ["1234567", "7654321", "1111111"]
        )
        assert result == [True, True, True]

    def test_partial_found(self):
        result = niah._score_multi_value(
            "1234567 and something", ["1234567", "7654321"]
        )
        assert result == [True, False]

    def test_none_found(self):
        result = niah._score_multi_value("no numbers here", ["1234567", "7654321"])
        assert result == [False, False]

    def test_rejects_6_digit_in_multi(self):
        """6-digit numbers should not match 7-digit expected values."""
        result = niah._score_multi_value("123456", ["1234567"])
        assert result == [False]


# ===================================================================
# Haystack assembly — single mode
# ===================================================================


class TestGenerateHaystackSingle:
    """Tests for generate_haystack_single()."""

    def test_needle_text_present(self):
        rng = random.Random(42)
        needle = niah.Needle(
            key="The special magic number is", value="1234567", depth_pct=0.5
        )
        haystack = niah.generate_haystack_single(needle, 4000, rng)
        assert needle.sentence in haystack

    def test_reaches_target_size(self):
        rng = random.Random(42)
        needle = niah.Needle(
            key="The special magic number is", value="1234567", depth_pct=0.5
        )
        haystack = niah.generate_haystack_single(needle, 4000, rng)
        assert len(haystack) >= 4000

    def test_needle_at_start(self):
        """Depth 0% should place needle near the beginning."""
        rng = random.Random(42)
        needle = niah.Needle(
            key="The special magic number is", value="1234567", depth_pct=0.0
        )
        haystack = niah.generate_haystack_single(needle, 4000, rng)
        assert needle.sentence in haystack
        # Should appear in the first quarter
        pos = haystack.index(needle.sentence)
        assert pos < len(haystack) * 0.25

    def test_needle_at_end(self):
        """Depth 100% should place needle near the end."""
        rng = random.Random(42)
        needle = niah.Needle(
            key="The special magic number is", value="1234567", depth_pct=1.0
        )
        haystack = niah.generate_haystack_single(needle, 4000, rng)
        assert needle.sentence in haystack
        pos = haystack.index(needle.sentence)
        assert pos > len(haystack) * 0.5


# ===================================================================
# Haystack assembly — multi-key mode
# ===================================================================


class TestGenerateHaystackMultiKey:
    """Tests for generate_haystack_multi_key()."""

    def test_real_and_distractors_present(self):
        rng = random.Random(42)
        real = niah.Needle(
            key="The special magic number is", value="1234567", depth_pct=0.5
        )
        distractors = [
            niah.Needle(key="The secret password is", value="7654321", depth_pct=0.25),
            niah.Needle(key="The hidden code is", value="1111111", depth_pct=0.75),
        ]
        haystack = niah.generate_haystack_multi_key(real, distractors, 4000, rng)
        assert real.sentence in haystack
        for d in distractors:
            assert d.sentence in haystack


# ===================================================================
# Haystack assembly — multi-value mode
# ===================================================================


class TestGenerateHaystackMultiValue:
    """Tests for generate_haystack_multi_value()."""

    def test_all_needles_present(self):
        rng = random.Random(42)
        needles = [
            niah.Needle(
                key="The special magic number is", value="1111111", depth_pct=0.25
            ),
            niah.Needle(
                key="The special magic number is", value="2222222", depth_pct=0.5
            ),
            niah.Needle(
                key="The special magic number is", value="3333333", depth_pct=0.75
            ),
        ]
        haystack = niah.generate_haystack_multi_value(needles, 4000, rng)
        for n in needles:
            assert n.sentence in haystack


# ===================================================================
# Insert needles at correct depth
# ===================================================================


class TestInsertNeedlesIntoParas:
    """Tests for _insert_needles_into_paragraphs()."""

    def test_needle_at_depth_zero(self):
        paragraphs = ["aaa", "bbb", "ccc", "ddd"]
        needle = niah.Needle(key="KEY", value="1234567", depth_pct=0.0)
        result = niah._insert_needles_into_paragraphs(paragraphs, [needle])
        parts = result.split("\n\n")
        assert parts[0] == needle.sentence

    def test_needle_at_depth_100(self):
        paragraphs = ["aaa", "bbb", "ccc", "ddd"]
        needle = niah.Needle(key="KEY", value="1234567", depth_pct=1.0)
        result = niah._insert_needles_into_paragraphs(paragraphs, [needle])
        parts = result.split("\n\n")
        assert parts[-1] == needle.sentence

    def test_multiple_needles_at_different_depths(self):
        paragraphs = [
            "aaa",
            "bbb",
            "ccc",
            "ddd",
            "eee",
            "fff",
            "ggg",
            "hhh",
            "iii",
            "jjj",
        ]
        n1 = niah.Needle(key="K1", value="1111111", depth_pct=0.0)
        n2 = niah.Needle(key="K2", value="2222222", depth_pct=1.0)
        result = niah._insert_needles_into_paragraphs(paragraphs, [n1, n2])
        parts = result.split("\n\n")
        assert parts[0] == n1.sentence
        assert parts[-1] == n2.sentence


# ===================================================================
# Data classes — ConfigResult, TrialResult
# ===================================================================


class TestDataClasses:
    """Tests for ConfigResult and TrialResult."""

    def test_config_result_accuracy_pct(self):
        cr = niah.ConfigResult(mode="single", context_length=4096, cache_type="q8_0")
        cr.trials = [
            niah.TrialResult(expected="1111111", response="1111111", found=True),
            niah.TrialResult(expected="2222222", response="nope", found=False),
            niah.TrialResult(expected="3333333", response="3333333", found=True),
        ]
        assert abs(cr.accuracy_pct - 66.666) < 1.0

    def test_config_result_passed_all_true(self):
        cr = niah.ConfigResult(mode="single", context_length=4096, cache_type="q8_0")
        cr.trials = [
            niah.TrialResult(expected="1111111", response="1111111", found=True),
            niah.TrialResult(expected="2222222", response="2222222", found=True),
        ]
        assert cr.passed is True

    def test_config_result_passed_one_false(self):
        cr = niah.ConfigResult(mode="single", context_length=4096, cache_type="q8_0")
        cr.trials = [
            niah.TrialResult(expected="1111111", response="1111111", found=True),
            niah.TrialResult(expected="2222222", response="wrong", found=False),
        ]
        assert cr.passed is False

    def test_config_result_empty_trials(self):
        cr = niah.ConfigResult(mode="single", context_length=4096, cache_type="q8_0")
        assert cr.accuracy_pct == 0.0
        assert cr.passed is True  # vacuously true (all() of empty)

    def test_config_result_all_hits(self):
        cr = niah.ConfigResult(mode="single", context_length=4096, cache_type="q8_0")
        cr.trials = [
            niah.TrialResult(expected="1111111", response="1111111", found=True),
        ]
        assert cr.accuracy_pct == 100.0

    def test_trial_result_defaults(self):
        t = niah.TrialResult(expected="1234567", response="1234567", found=True)
        assert t.needle_depth_pct == 0.0
        assert t.context_length == 0


# ===================================================================
# Heatmap table generation (single mode)
# ===================================================================


class TestHeatmapTable:
    """Tests for _build_heatmap_table()."""

    def test_basic_heatmap(self):
        results = [
            niah.ConfigResult(
                mode="single",
                context_length=4096,
                cache_type="q8_0",
                needle_depth_pct=0.5,
            ),
        ]
        results[0].trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]
        table = niah._build_heatmap_table(results, "q8_0", "test-model")
        assert "q8_0" in table
        assert "50" in table  # depth 50%

    def test_no_results(self):
        table = niah._build_heatmap_table([], "q8_0", "test-model")
        assert "no results" in table

    def test_err_for_no_trials(self):
        results = [
            niah.ConfigResult(
                mode="single",
                context_length=4096,
                cache_type="q8_0",
                needle_depth_pct=0.5,
            ),
        ]
        # No trials — passed is vacuously True, but we still get a row
        table = niah._build_heatmap_table(results, "q8_0", "test-model")
        # Should have a row with depth 50%
        assert "50" in table


# ===================================================================
# Delta table
# ===================================================================


class TestDeltaTable:
    """Tests for _build_delta_table()."""

    def test_delta_with_difference(self):
        r1 = niah.ConfigResult(
            mode="single", context_length=4096, cache_type="q8_0", needle_depth_pct=0.5
        )
        r1.trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]
        r2 = niah.ConfigResult(
            mode="single",
            context_length=4096,
            cache_type="turbo3",
            needle_depth_pct=0.5,
        )
        r2.trials = [
            niah.TrialResult(expected="1234567", response="wrong", found=False)
        ]
        table = niah._build_delta_table([r1, r2], "q8_0", "turbo3")
        assert "Delta" in table

    def test_delta_empty_when_no_baseline(self):
        table = niah._build_delta_table([], "q8_0", "turbo3")
        assert table == ""


# ===================================================================
# Multi-key table
# ===================================================================


class TestMultiKeyTable:
    """Tests for _build_multi_key_table()."""

    def test_basic_multi_key_table(self):
        r = niah.ConfigResult(
            mode="multi-key",
            context_length=4096,
            cache_type="q8_0",
            needle_depth_pct=0.5,
        )
        r.trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]
        table = niah._build_multi_key_table([r], "test-model")
        assert "Multi-Key" in table
        assert "q8_0" in table


# ===================================================================
# Multi-value table
# ===================================================================


class TestMultiValueTable:
    """Tests for _build_multi_value_table()."""

    def test_basic_multi_value_table(self):
        r = niah.ConfigResult(
            mode="multi-value", context_length=4096, cache_type="q8_0", needle_count=2
        )
        r.trials = [
            niah.TrialResult(
                expected="1111111", response="1111111, 2222222", found=True
            ),
            niah.TrialResult(
                expected="2222222", response="1111111, 2222222", found=True
            ),
        ]
        table = niah._build_multi_value_table([r], "test-model")
        assert "Multi-Value" in table


# ===================================================================
# build_output (all three modes)
# ===================================================================


class TestBuildOutput:
    """Tests for build_output()."""

    def test_single_mode(self):
        r = niah.ConfigResult(
            mode="single", context_length=4096, cache_type="q8_0", needle_depth_pct=0.5
        )
        r.trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]
        output = niah.build_output([r], "test-model", "single")
        assert "test-model" in output
        assert "single" in output.lower()

    def test_multi_key_mode(self):
        r = niah.ConfigResult(mode="multi-key", context_length=4096, cache_type="q8_0")
        r.trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]
        output = niah.build_output([r], "test-model", "multi-key")
        assert "Multi-Key" in output

    def test_multi_value_mode(self):
        r = niah.ConfigResult(
            mode="multi-value", context_length=4096, cache_type="q8_0", needle_count=2
        )
        r.trials = [
            niah.TrialResult(
                expected="1111111", response="1111111, 2222222", found=True
            ),
            niah.TrialResult(
                expected="2222222", response="1111111, 2222222", found=True
            ),
        ]
        output = niah.build_output([r], "test-model", "multi-value")
        assert "Multi-Value" in output

    def test_single_mode_with_delta(self):
        """Two cache types in single mode should produce a delta table."""
        r1 = niah.ConfigResult(
            mode="single", context_length=4096, cache_type="q8_0", needle_depth_pct=0.5
        )
        r1.trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]
        r2 = niah.ConfigResult(
            mode="single",
            context_length=4096,
            cache_type="turbo3",
            needle_depth_pct=0.5,
        )
        r2.trials = [
            niah.TrialResult(expected="1234567", response="wrong", found=False)
        ]
        output = niah.build_output([r1, r2], "test-model", "single")
        assert "Delta" in output


# ===================================================================
# Port finding
# ===================================================================


class TestFindFreePort:
    """Tests for _find_free_port()."""

    def test_finds_port(self):
        port = niah._find_free_port(18000)
        assert 18000 <= port < 18100

    def test_port_conflict_skips_busy(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 19000))
            s.listen(1)
            port = niah._find_free_port(19000)
            assert port != 19000
            assert 19001 <= port < 19100

    def test_all_ports_busy_raises(self):
        def always_busy(*args, **kwargs):
            s = MagicMock()
            s.__enter__ = MagicMock(return_value=s)
            s.__exit__ = MagicMock(return_value=False)
            s.bind = MagicMock(side_effect=OSError("Address in use"))
            return s

        with patch("socket.socket", side_effect=always_busy):
            with pytest.raises(RuntimeError, match="No free port"):
                niah._find_free_port(8090)


# ===================================================================
# Server management
# ===================================================================


class TestStartServer:
    """Tests for start_server()."""

    def _make_health_response(self, status="ok"):
        body = json.dumps({"status": status}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def _make_proc(self, poll_return=None, returncode=0):
        proc = MagicMock()
        proc.poll.return_value = poll_return
        proc.returncode = returncode
        return proc

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_successful_startup(self, mock_popen, mock_urlopen, mock_sleep):
        proc = self._make_proc()
        mock_popen.return_value = proc
        mock_urlopen.return_value = self._make_health_response("ok")

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()
            server_bin.chmod(0o755)

            result = niah.start_server(
                Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090
            )
            assert result is proc

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_server_binary_not_found(self, mock_popen, mock_urlopen, mock_sleep):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError, match="llama-server not found"):
                niah.start_server(
                    Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090
                )

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_server_exits_prematurely(self, mock_popen, mock_urlopen, mock_sleep):
        proc = self._make_proc(poll_return=1, returncode=1)
        mock_popen.return_value = proc

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            with pytest.raises(RuntimeError, match="exited prematurely"):
                niah.start_server(
                    Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090
                )

    @patch("time.monotonic")
    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_server_health_timeout(
        self, mock_popen, mock_urlopen, mock_sleep, mock_monotonic
    ):
        proc = self._make_proc()
        mock_popen.return_value = proc
        mock_monotonic.side_effect = [0, 0, 121]
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            with pytest.raises(TimeoutError, match="did not become healthy"):
                niah.start_server(
                    Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090
                )

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_health_check_retries_on_url_error(
        self, mock_popen, mock_urlopen, mock_sleep
    ):
        proc = self._make_proc()
        mock_popen.return_value = proc
        mock_urlopen.side_effect = [
            urllib.error.URLError("refused"),
            urllib.error.URLError("refused"),
            self._make_health_response("ok"),
        ]

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            result = niah.start_server(
                Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090
            )
            assert result is proc


class TestStopServer:
    """Tests for stop_server()."""

    def test_graceful_stop(self):
        proc = MagicMock()
        niah.stop_server(proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=10)

    def test_force_kill_on_terminate_failure(self):
        proc = MagicMock()
        proc.terminate.side_effect = Exception("won't stop")
        niah.stop_server(proc)
        proc.kill.assert_called_once()


class TestCleanupServer:
    """Tests for _cleanup_server()."""

    def test_cleanup_terminates(self):
        proc = MagicMock()
        niah._active_server = proc
        niah._cleanup_server()
        proc.terminate.assert_called_once()
        assert niah._active_server is None

    def test_cleanup_kills_on_terminate_timeout(self):
        proc = MagicMock()
        proc.wait.side_effect = Exception("timeout")
        niah._active_server = proc
        niah._cleanup_server()
        proc.kill.assert_called_once()
        assert niah._active_server is None

    def test_cleanup_noop_when_none(self):
        niah._active_server = None
        niah._cleanup_server()  # Should not raise


# ===================================================================
# Query logic
# ===================================================================


class TestQueryServer:
    """Tests for _query_server()."""

    def _make_chat_response(self, content: str):
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_successful_query(self, mock_urlopen, mock_sleep):
        mock_urlopen.return_value = self._make_chat_response("1234567")
        result = niah._query_server(8090, "some haystack")
        assert result == "1234567"

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_retries_on_network_error(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            urllib.error.URLError("timeout"),
            urllib.error.URLError("timeout"),
            self._make_chat_response("6543210"),
        ]
        result = niah._query_server(8090, "haystack", max_retries=3)
        assert result == "6543210"

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_all_retries_exhausted(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        with pytest.raises(RuntimeError, match="Failed to query server"):
            niah._query_server(8090, "haystack", max_retries=3)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_garbage_json_response(self, mock_urlopen, mock_sleep):
        resp = MagicMock()
        resp.read.return_value = b"not json at all"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            niah._query_server(8090, "haystack")

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_response_stripped(self, mock_urlopen, mock_sleep):
        mock_urlopen.return_value = self._make_chat_response("  1234567  \n")
        result = niah._query_server(8090, "haystack")
        assert result == "1234567"

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_thinking_tags_stripped(self, mock_urlopen, mock_sleep):
        mock_urlopen.return_value = self._make_chat_response(
            "<think>let me think...</think>1234567"
        )
        result = niah._query_server(8090, "haystack")
        assert result == "1234567"


# ===================================================================
# Save results
# ===================================================================


class TestSaveResults:
    """Tests for save_results()."""

    def test_creates_json_and_md(self):
        r = niah.ConfigResult(
            mode="single", context_length=4096, cache_type="q8_0", needle_depth_pct=0.5
        )
        r.trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]

        with tempfile.TemporaryDirectory() as td:
            json_path, md_path = niah.save_results(
                [r], "test-model", "single", Path(td)
            )
            assert json_path.exists()
            assert md_path.exists()
            assert json_path.suffix == ".json"
            assert md_path.suffix == ".md"

            with open(json_path) as f:
                data = json.load(f)
            assert data["model"] == "test-model"
            assert data["mode"] == "single"
            assert data["seed"] == 42
            assert len(data["results"]) == 1

    def test_creates_output_dir(self):
        r = niah.ConfigResult(mode="single", context_length=4096, cache_type="q8_0")
        r.trials = []

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "nested" / "output"
            niah.save_results([r], "model", "single", out)
            assert out.exists()

    def test_no_home_dir_in_output(self):
        """Output paths should not contain /Users/dipesh/."""
        r = niah.ConfigResult(mode="single", context_length=4096, cache_type="q8_0")
        r.trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]

        with tempfile.TemporaryDirectory() as td:
            json_path, md_path = niah.save_results(
                [r], "test-model", "single", Path(td)
            )
            with open(md_path) as f:
                content = f.read()
            assert "/Users/dipesh/" not in content


# ===================================================================
# Argparse
# ===================================================================


class TestParseArgs:
    """Tests for parse_args()."""

    def test_required_arg(self):
        args = niah.parse_args(["/path/to/llama"])
        assert args.llama_dir == "/path/to/llama"

    def test_both_positional_args(self):
        args = niah.parse_args(["/llama", "/model.gguf"])
        assert args.llama_dir == "/llama"
        assert args.model_path == "/model.gguf"

    def test_defaults(self):
        args = niah.parse_args(["/llama"])
        assert args.depths == "4096,8192,16384,32768"
        assert args.mode == "single"
        assert args.port == "8090"
        assert args.cache_types == "q8_0,turbo3"
        assert args.verbose is False
        assert args.output_dir is None
        assert args.num_distractors == 3
        assert args.value_counts == "2,4,8"
        assert args.depths_sweep == "0,10,20,30,40,50,60,70,80,90,100"
        assert args.server_timeout == 120
        assert args.query_timeout == 300
        assert args.server_bin is None
        assert args.chars_per_token == 4.0

    def test_mode_single(self):
        args = niah.parse_args(["/llama", "--mode", "single"])
        assert args.mode == "single"

    def test_mode_multi_key(self):
        args = niah.parse_args(["/llama", "--mode", "multi-key"])
        assert args.mode == "multi-key"

    def test_mode_multi_value(self):
        args = niah.parse_args(["/llama", "--mode", "multi-value"])
        assert args.mode == "multi-value"

    def test_depths_sweep(self):
        args = niah.parse_args(["/llama", "--depths-sweep", "0,50,100"])
        assert args.depths_sweep == "0,50,100"

    def test_num_distractors(self):
        args = niah.parse_args(["/llama", "--num-distractors", "5"])
        assert args.num_distractors == 5

    def test_value_counts(self):
        args = niah.parse_args(["/llama", "--value-counts", "1,2,3"])
        assert args.value_counts == "1,2,3"

    def test_verbose_flag(self):
        args = niah.parse_args(["/llama", "-v"])
        assert args.verbose is True

    def test_all_options(self):
        args = niah.parse_args(
            [
                "/llama",
                "/model.gguf",
                "--mode",
                "multi-key",
                "--depths",
                "1024",
                "--depths-sweep",
                "0,50,100",
                "--cache-types",
                "f16",
                "--num-distractors",
                "7",
                "--value-counts",
                "1,2",
                "--port",
                "9090",
                "--output-dir",
                "/tmp/out",
                "--verbose",
                "--server-timeout",
                "60",
                "--query-timeout",
                "120",
                "--server-bin",
                "/path/to/server",
                "--chars-per-token",
                "3.5",
            ]
        )
        assert args.mode == "multi-key"
        assert args.depths == "1024"
        assert args.depths_sweep == "0,50,100"
        assert args.cache_types == "f16"
        assert args.num_distractors == 7
        assert args.value_counts == "1,2"
        assert args.port == "9090"
        assert args.output_dir == "/tmp/out"
        assert args.verbose is True
        assert args.server_timeout == 60
        assert args.query_timeout == 120
        assert args.server_bin == "/path/to/server"
        assert args.chars_per_token == 3.5

    def test_missing_required_arg(self):
        with pytest.raises(SystemExit):
            niah.parse_args([])


# ===================================================================
# Signal handler
# ===================================================================


class TestSignalHandler:
    """Tests for _signal_handler()."""

    @patch.object(niah, "_cleanup_server")
    def test_signal_handler_calls_cleanup_and_exits(self, mock_cleanup):
        with pytest.raises(SystemExit) as exc_info:
            niah._signal_handler(2, None)
        mock_cleanup.assert_called_once()
        assert exc_info.value.code == 1


# ===================================================================
# main() end-to-end (fully mocked)
# ===================================================================


class TestMain:
    """Tests for main()."""

    @patch.object(
        niah,
        "save_results",
        return_value=(Path("/fake/results.json"), Path("/fake/results.md")),
    )
    @patch.object(niah, "build_output", return_value="output text")
    @patch.object(niah, "run_single_mode", return_value=[])
    def test_main_single_mode(self, mock_run, mock_output, mock_save):
        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.main([str(llama_dir), str(model_path), "--mode", "single"])
            mock_run.assert_called_once()

    @patch.object(
        niah,
        "save_results",
        return_value=(Path("/fake/results.json"), Path("/fake/results.md")),
    )
    @patch.object(niah, "build_output", return_value="output text")
    @patch.object(niah, "run_multi_key_mode", return_value=[])
    def test_main_multi_key_mode(self, mock_run, mock_output, mock_save):
        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.main([str(llama_dir), str(model_path), "--mode", "multi-key"])
            mock_run.assert_called_once()

    @patch.object(
        niah,
        "save_results",
        return_value=(Path("/fake/results.json"), Path("/fake/results.md")),
    )
    @patch.object(niah, "build_output", return_value="output text")
    @patch.object(niah, "run_multi_value_mode", return_value=[])
    def test_main_multi_value_mode(self, mock_run, mock_output, mock_save):
        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.main([str(llama_dir), str(model_path), "--mode", "multi-value"])
            mock_run.assert_called_once()

    def test_main_missing_model_path(self):
        """Should sys.exit if model_path not provided."""
        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            with pytest.raises(SystemExit):
                niah.main([str(llama_dir)])

    @patch.object(
        niah,
        "save_results",
        return_value=(Path("/fake/results.json"), Path("/fake/results.md")),
    )
    @patch.object(niah, "build_output", return_value="output text")
    @patch.object(niah, "run_single_mode", return_value=[])
    def test_main_default_output_dir(self, mock_run, mock_output, mock_save):
        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.main([str(llama_dir), str(model_path)])
            call_args = mock_save.call_args
            # output_dir is the 4th positional arg
            expected = Path(niah.__file__).resolve().parents[2] / "artifacts" / "niah"
            assert Path(call_args[0][3]) == expected

    @patch.object(
        niah,
        "save_results",
        return_value=(Path("/fake/results.json"), Path("/fake/results.md")),
    )
    @patch.object(niah, "build_output", return_value="output text")
    @patch.object(niah, "run_single_mode", return_value=[])
    def test_main_custom_output_dir(self, mock_run, mock_output, mock_save):
        with tempfile.TemporaryDirectory() as td:
            llama_dir = Path(td)
            model_path = llama_dir / "model.gguf"
            model_path.touch()
            server_bin = llama_dir / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.main(
                [str(llama_dir), str(model_path), "--output-dir", "/tmp/custom_out"]
            )
            call_args = mock_save.call_args
            assert call_args[0][3] == Path("/tmp/custom_out")


# ===================================================================
# Verbose mode in start_server
# ===================================================================


class TestStartServerVerbose:
    """Test verbose output paths in start_server()."""

    def _make_health_response(self):
        body = json.dumps({"status": "ok"}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_verbose_prints_cmd(self, mock_popen, mock_urlopen, mock_sleep, capsys):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        mock_urlopen.return_value = self._make_health_response()

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.start_server(
                Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090, verbose=True
            )

        captured = capsys.readouterr()
        assert "[CMD]" in captured.out

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    @patch("subprocess.Popen")
    def test_verbose_passes_stdout_stderr(self, mock_popen, mock_urlopen, mock_sleep):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        mock_urlopen.return_value = self._make_health_response()

        with tempfile.TemporaryDirectory() as td:
            server_bin = Path(td) / "build" / "bin" / "llama-server"
            server_bin.parent.mkdir(parents=True)
            server_bin.touch()

            niah.start_server(
                Path(td), Path("/fake/model.gguf"), "q8_0", 4096, 8090, verbose=True
            )

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["stdout"] is None
        assert call_kwargs["stderr"] is None


# ===================================================================
# No /Users/dipesh/ in output
# ===================================================================


class TestNoHomeDir:
    """Ensure no hardcoded /Users/dipesh/ paths leak into test output."""

    def test_filler_no_home_dir(self):
        rng = random.Random(42)
        paragraphs = niah._build_filler(4000, rng)
        for p in paragraphs:
            assert "/Users/dipesh/" not in p

    def test_needle_sentence_no_home_dir(self):
        n = niah.Needle(
            key="The special magic number is", value="1234567", depth_pct=0.5
        )
        assert "/Users/dipesh/" not in n.sentence

    def test_heatmap_no_home_dir(self):
        r = niah.ConfigResult(
            mode="single", context_length=4096, cache_type="q8_0", needle_depth_pct=0.5
        )
        r.trials = [
            niah.TrialResult(expected="1234567", response="1234567", found=True)
        ]
        table = niah._build_heatmap_table([r], "q8_0", "test-model")
        assert "/Users/dipesh/" not in table
