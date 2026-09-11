"""Metria core models for reproducible inference-systems studies."""

from .artifacts import (
    ArtifactIntegrityError,
    artifact_to_data,
    extract_verified_zip,
    fetch_artifact,
    verify_artifact,
)
from .capabilities import (
    GeometryInspection,
    ModelGeometry,
    inspect_model_geometry,
)
from .capability_checks import (
    CapabilityCheck,
    CapabilityCheckRegistry,
    CapabilityCheckResult,
)
from .comparison import compare_runs
from .execution import execute_run
from .hardware import capture_hardware_fingerprint
from .identity import (
    ArtifactManifest,
    Capability,
    CapabilitySet,
    HardwareFingerprint,
    ModelRef,
    RuntimeConfig,
    SupportLevel,
    WorkloadSpec,
)
from .identity_evidence import (
    IdentityStatus,
    RuntimeIdentityEvidence,
    aggregate_identity_status,
)
from .inspection import (
    PreflightCapabilityResult,
    capability_inspection_to_mapping,
    inspect_run_capabilities,
)
from .integrations.turboquant import evaluate_turboquant_kv_capability
from .models import (
    ComparisonPlan,
    CompatibilityIssue,
    CompatibilityReport,
    MetricDefinition,
    MetricDirection,
    MetricSample,
    MetricSummary,
    RunRecord,
    RunSpec,
    RunStatus,
    StudySpec,
    TreatmentSpec,
    TreatmentType,
)
from .policies import (
    PolicyCriterion,
    PolicyDecision,
    PolicyEvaluation,
    VerificationPolicy,
    policy_from_data,
)
from .processes import ProcessError, ProcessResult, run_process
from .protocols import MeasurementResult
from .recipes import (
    StudyRecipe,
    dump_study_recipe,
    load_study_recipe,
    run_spec_from_data,
    run_spec_to_data,
    study_recipe_digest,
    study_recipe_from_data,
    study_recipe_to_data,
    study_recipe_to_json,
)
from .records import (
    RUN_RECORD_SCHEMA,
    dump_run_record,
    load_run_record,
    run_evidence_digest,
    run_record_digest,
    run_record_from_data,
    run_record_to_data,
    run_record_to_json,
)
from .study_execution import (
    PairwiseAnalysisStatus,
    StudyExecutionResult,
    StudyPairAnalysis,
    StudyPairComparison,
    execute_study,
)

__version__ = "0.1.1.dev0"

from .verification import VerificationResult, VerificationVerdict, verify_recipe

__all__ = [
    "RUN_RECORD_SCHEMA",
    "ArtifactManifest",
    "ArtifactIntegrityError",
    "Capability",
    "CapabilitySet",
    "CapabilityCheck",
    "CapabilityCheckRegistry",
    "CapabilityCheckResult",
    "ComparisonPlan",
    "CompatibilityIssue",
    "CompatibilityReport",
    "GeometryInspection",
    "HardwareFingerprint",
    "IdentityStatus",
    "MeasurementResult",
    "MetricDefinition",
    "MetricDirection",
    "MetricSample",
    "MetricSummary",
    "ModelGeometry",
    "ModelRef",
    "PairwiseAnalysisStatus",
    "PreflightCapabilityResult",
    "PolicyCriterion",
    "PolicyDecision",
    "PolicyEvaluation",
    "VerificationPolicy",
    "ProcessError",
    "ProcessResult",
    "RunRecord",
    "RunSpec",
    "RunStatus",
    "RuntimeConfig",
    "RuntimeIdentityEvidence",
    "StudyExecutionResult",
    "StudyPairAnalysis",
    "StudyPairComparison",
    "StudyRecipe",
    "StudySpec",
    "SupportLevel",
    "TreatmentSpec",
    "TreatmentType",
    "WorkloadSpec",
    "VerificationResult",
    "VerificationVerdict",
    "aggregate_identity_status",
    "artifact_to_data",
    "extract_verified_zip",
    "fetch_artifact",
    "verify_artifact",
    "capability_inspection_to_mapping",
    "capture_hardware_fingerprint",
    "compare_runs",
    "dump_run_record",
    "dump_study_recipe",
    "evaluate_turboquant_kv_capability",
    "execute_run",
    "execute_study",
    "inspect_model_geometry",
    "inspect_run_capabilities",
    "load_run_record",
    "load_study_recipe",
    "run_evidence_digest",
    "run_record_digest",
    "run_process",
    "run_record_from_data",
    "run_record_to_data",
    "run_record_to_json",
    "run_spec_from_data",
    "run_spec_to_data",
    "study_recipe_digest",
    "study_recipe_from_data",
    "study_recipe_to_data",
    "study_recipe_to_json",
    "verify_recipe",
    "policy_from_data",
]
