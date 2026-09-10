from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from runtime_contract import RuntimeContractCase, exercise_runtime_contract

from metria import RunSpec
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    SupportReport,
)


class _ContractSession:
    def __init__(self, runtime_name: str) -> None:
        self.runtime_name = runtime_name
        self.closed = False
        self.reset_count = 0
        self.calls = 0

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        if self.closed:
            raise RuntimeError("runtime session is closed")
        self.calls += 1
        return InferenceBatch(
            outputs=tuple(f"output-{index}" for index, _ in enumerate(requests)),
            captures={item.kind: () for item in capture},
            metadata={"runtime": self.runtime_name, "request_count": len(requests)},
        )

    def reset(self, scope: str = "measurement") -> None:
        if self.closed:
            raise RuntimeError("runtime session is closed")
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class _ContractAdapter:
    name = "contract-fake"

    def __init__(self) -> None:
        self.launch_count = 0
        self.capture_probe_count = 0

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        return SupportReport(
            status="supported",
            evidence={
                "runtime": self.name,
                "requested_model": spec.model.get("id"),
                "environment_class": environment.get("hardware_class"),
            },
        )

    def probe_captures(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
        capture: Sequence[CaptureRequest],
    ) -> SupportReport:
        del spec, environment
        self.capture_probe_count += 1
        kinds = tuple(item.kind for item in capture)
        if kinds != ("token_ids",):
            return SupportReport(
                status="unsupported",
                reasons=("contract fixture supports only token_ids",),
            )
        return SupportReport(
            status="supported",
            evidence={"token_ids": "fixture_native_capture"},
        )

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "runtime": {"name": self.name, "version": "1"},
            "model": dict(spec.model),
            "environment": dict(environment),
        }

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> _ContractSession:
        del resolved, environment
        self.launch_count += 1
        return _ContractSession(self.name)

    def observe(self, session: _ContractSession) -> Mapping[str, Any]:
        return {
            "runtime": self.name,
            "state": "confirmed",
            "reset_count": session.reset_count,
            "inference_calls": session.calls,
        }


def _spec() -> RunSpec:
    return RunSpec(
        model={"id": "example/model", "revision": "abc123"},
        runtime={"name": "contract-fake"},
        scenario={"name": "decode", "max_tokens": 4},
        measurements=("contract.measurement",),
    )


def test_reusable_runtime_contract_harness() -> None:
    adapter = _ContractAdapter()
    case = RuntimeContractCase(
        adapter=adapter,
        spec=_spec(),
        environment={"hardware_class": "test"},
        requests=(
            InferenceRequest(prompt="metria contract prompt"),
            InferenceRequest(prompt="second private prompt"),
        ),
        capture=(CaptureRequest(kind="token_ids"),),
        privacy_terms=("metria contract prompt", "second private prompt"),
    )

    exercise_runtime_contract(case)

    assert adapter.capture_probe_count == 1
    assert adapter.launch_count == 1
