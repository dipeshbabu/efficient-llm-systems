"""Qualify a local capture provider and write a CPU-thread comparison recipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metria import (
    ComparisonPlan,
    RunSpec,
    RunStatus,
    StudyRecipe,
    StudySpec,
    execute_run,
)
from metria.measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from metria.recipes import study_recipe_to_json
from metria.records import run_record_to_json
from metria.runtimes.llamacpp import LlamaCppAdapter, _find_binary, _sha256_file

TINY_MODEL_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", default=TINY_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=Path("study.json"))
    parser.add_argument("--reference-threads", type=int, default=1)
    parser.add_argument("--candidate-threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    qualification = output.with_suffix(".qualification.run.json")
    if output.exists() or qualification.exists():
        parser.error(
            "choose new output and qualification paths; existing evidence is preserved"
        )
    if args.reference_threads <= 0 or args.candidate_threads <= 0:
        parser.error("thread counts must be positive")
    bin_dir = args.bin_dir.expanduser().resolve()
    binary = _find_binary(bin_dir, "llama-completion")
    if binary is None:
        parser.error("bin-dir must contain the patched llama-completion provider")
    environment = {"llama_cpp_token_ids_capture_sha256": _sha256_file(binary)}
    runs = tuple(
        RunSpec(
            model={
                "path": str(args.model.expanduser().resolve()),
                "sha256": args.model_sha256,
            },
            runtime={
                "name": "llamacpp",
                "bin_dir": str(bin_dir),
                "n_gpu_layers": 0,
                "flash_attention": False,
                "threads": threads,
                "threads_batch": 1,
            },
            scenario={
                "context": 256,
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 42,
                "chat_template": False,
            },
            measurements=(TokenTrajectoryProtocol.name,),
        )
        for threads in (args.reference_threads, args.candidate_threads)
    )
    record = execute_run(
        study_name="llamacpp-capture-qualification",
        run_id="qualification",
        spec=runs[0],
        adapter=LlamaCppAdapter(),
        measurement=TokenTrajectoryProtocol(),
        measurement_config={
            "prompts": [{"id": "probe", "prompt": "Once upon a time"}],
            "generation": {"max_tokens": 1},
        },
        environment=environment,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with qualification.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(run_record_to_json(record) + "\n")
    identity = record.observed.get("identity", {})
    fields = identity.get("applied", {}).get("fields", {})
    valid = (
        record.status is RunStatus.COMPLETED
        and identity.get("model", {}).get("sha256") == args.model_sha256.lower()
        and fields.get("threads") == args.reference_threads
        and fields.get("threads_batch") == 1
        and fields.get("context") == 256
        and fields.get("chat_template_applied") is False
        and record.metrics.get("trajectory_nonempty_capture_rate") is not None
        and record.metrics["trajectory_nonempty_capture_rate"].value == 1.0
    )
    if not valid:
        print(f"Capture qualification failed; inspect {qualification}", file=sys.stderr)
        return 1
    recipe = StudyRecipe(
        study=StudySpec(
            name="llamacpp-cpu-thread-change",
            runs=runs,
            comparison=ComparisonPlan(
                vary=frozenset(
                    {
                        "runtime.threads",
                        "resolved.runtime.threads",
                        "observed.runtime.threads",
                        "observed.identity.applied.fields.threads",
                    }
                ),
                control=frozenset({"model", "scenario", "measurements"}),
                analyses=(TrajectoryAgreementAnalysis.name,),
            ),
        ),
        measurement_configs={
            TokenTrajectoryProtocol.name: {
                "prompts": [
                    {"id": "story", "prompt": "Once upon a time"},
                    {"id": "dog", "prompt": "The little dog"},
                    {"id": "sun", "prompt": "The sun was"},
                ]
            }
        },
        environment=environment,
    )
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(study_recipe_to_json(recipe) + "\n")
    print(f"Recipe: {output}")
    print(f"Provider qualification: {qualification}")
    print(
        f"Capture provider SHA-256: {environment['llama_cpp_token_ids_capture_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
