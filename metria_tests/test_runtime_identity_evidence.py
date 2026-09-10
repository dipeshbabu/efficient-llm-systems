from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from metria import (
    RunRecord,
    RunSpec,
    RunStatus,
    run_record_digest,
    run_record_from_data,
    run_record_to_json,
)
from metria.identity_evidence import (
    IdentityStatus,
    RuntimeIdentityEvidence,
    aggregate_identity_status,
)


def _verified_component(**values: object) -> dict[str, object]:
    return {"status": "verified", **values}


def test_identity_status_aggregates_conservatively() -> None:
    assert aggregate_identity_status(()) is IdentityStatus.UNKNOWN
    assert (
        aggregate_identity_status(("verified", "verified")) is IdentityStatus.VERIFIED
    )
    assert aggregate_identity_status(("verified", "unknown")) is IdentityStatus.PARTIAL
    assert aggregate_identity_status(("partial", "verified")) is IdentityStatus.PARTIAL
    assert (
        aggregate_identity_status(("verified", "mismatch")) is IdentityStatus.MISMATCH
    )


def test_runtime_identity_is_mapping_compatible_and_deeply_immutable() -> None:
    source = {"status": "verified", "id": "example/model", "nested": {"x": 1}}
    identity = RuntimeIdentityEvidence(
        status="verified",
        model=source,
        tokenizer=_verified_component(id="example/model"),
        runtime=_verified_component(name="vllm", version="1.2.3"),
        chat_template=_verified_component(sha256="a" * 64),
        applied=_verified_component(fields={"cache.dtype": "fp8"}),
    )
    source["nested"]["x"] = 2

    assert identity["schema"] == "metria.runtime_identity.v1"
    assert identity["status"] == "verified"
    assert identity["model"]["nested"]["x"] == 1
    assert isinstance(identity.to_mapping(), MappingProxyType)
    with pytest.raises(TypeError):
        identity["runtime"]["name"] = "changed"


def test_mapping_views_keep_key_order_and_hide_non_mapping_attributes() -> None:
    identity = RuntimeIdentityEvidence(
        status=IdentityStatus.VERIFIED,
        model={"status": IdentityStatus.VERIFIED, "id": "example/model"},
    )
    expected_keys = (
        "schema",
        "status",
        "model",
        "tokenizer",
        "runtime",
        "chat_template",
        "applied",
        "endpoint",
        "reasons",
    )
    assert tuple(identity) == expected_keys
    assert tuple(identity.to_mapping()) == expected_keys
    assert len(identity) == len(expected_keys)
    assert dict(identity) == identity.to_mapping()
    assert type(identity["status"]) is str
    assert type(identity["model"]["status"]) is str
    for key in ("missing", "__dict__", "to_mapping"):
        assert key not in identity
        assert identity.get(key, "fallback") == "fallback"
        with pytest.raises(KeyError):
            identity[key]
    with pytest.raises(TypeError):
        identity[[]]


def test_all_mapping_views_retain_detached_nested_evidence() -> None:
    values = [{"threads": [1, 2]}]
    identity = RuntimeIdentityEvidence(
        status="verified", applied=_verified_component(samples=values)
    )
    views = [identity, identity.to_mapping(), dict(identity), dict(identity.items())]
    values[0]["threads"].append(3)
    for view in views:
        assert view["applied"]["samples"][0]["threads"] == (1, 2)
        with pytest.raises(TypeError):
            view["applied"]["samples"][0]["threads"] = ()
    with pytest.raises(TypeError):
        identity.to_mapping()["status"] = "unknown"


def test_runtime_identity_cannot_overstate_component_authority() -> None:
    with pytest.raises(ValueError, match="aggregate component authority"):
        RuntimeIdentityEvidence(
            status="verified",
            model=_verified_component(id="example/model"),
            tokenizer={"status": "unknown"},
        )

    with pytest.raises(ValueError, match="aggregate component authority"):
        RuntimeIdentityEvidence(status="verified")

    identity = RuntimeIdentityEvidence(
        status="partial",
        model=_verified_component(id="example/model"),
        tokenizer={"status": "unknown", "source": "unavailable"},
    )
    assert identity["status"] == "partial"


def test_runtime_identity_rejects_malformed_component_and_reasons() -> None:
    with pytest.raises(TypeError, match="model identity must be a mapping"):
        RuntimeIdentityEvidence(status="unknown", model="model")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model identity must declare status"):
        RuntimeIdentityEvidence(status="unknown", model={"id": "example/model"})
    with pytest.raises(TypeError, match="reasons must be a sequence"):
        RuntimeIdentityEvidence(status="unknown", reasons="reason")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        RuntimeIdentityEvidence(status="unknown", reasons=("",))


def test_endpoint_identity_rejects_sensitive_or_arbitrary_fields() -> None:
    safe = RuntimeIdentityEvidence(
        status="unknown",
        endpoint={
            "scheme": "https",
            "host": "inference.internal",
            "port": 443,
            "path_sha256": "b" * 64,
        },
    )
    assert safe["endpoint"]["host"] == "inference.internal"

    for field in ("authorization", "api_key", "token", "url"):
        with pytest.raises(ValueError, match="potentially sensitive"):
            RuntimeIdentityEvidence(
                status="unknown",
                endpoint={field: "secret-value"},
            )


def test_runtime_identity_round_trips_through_run_record_serialization() -> None:
    identity = RuntimeIdentityEvidence(
        status="partial",
        model=_verified_component(identifier="example/model", revision="abc123"),
        tokenizer={"status": "unknown"},
        runtime=_verified_component(name="vllm", version="1.2.3"),
    )
    record = RunRecord(
        study_name="identity-round-trip",
        run_id="candidate",
        requested=RunSpec(
            model={"id": "example/model", "revision": "abc123"},
            runtime={"name": "vllm", "version": "1.2.3"},
            scenario={"name": "decode"},
            measurements=("identity",),
        ),
        resolved={},
        observed={"identity": identity.to_mapping()},
        status=RunStatus.COMPLETED,
    )

    payload = run_record_to_json(record)
    restored = run_record_from_data(json.loads(payload))

    assert restored.observed["identity"] == identity.to_mapping()
    assert run_record_digest(restored) == run_record_digest(record)


def test_invalid_identity_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        RuntimeIdentityEvidence(status="trusted")
