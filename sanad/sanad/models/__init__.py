"""Typed contracts exchanged between Sanad components."""

from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.analysis import (
    AnalysisBundle,
    AnalysisFinding,
    AnalysisStatus,
    ContractAnalysisResult,
    CvAnalysisResult,
    EvidenceReference,
    FindingStatus,
    TargetJob,
)
from sanad.models.comparison import (
    ComparedContract,
    ContractComparisonResult,
    DimensionComparison,
    Recommendation,
)
from sanad.models.orchestration import (
    DocumentRole,
    OrchestratorResult,
    Route,
    SanadRequest,
    TaskHint,
    UploadedDocument,
)
from sanad.models.documents import (
    DocumentPage,
    DocumentSection,
    DocumentStatus,
    DocumentTable,
    FileType,
    ParsedDocument,
)
from sanad.models.extraction import (
    ContractExtraction,
    CvExtraction,
    ExtractedField,
    FieldStatus,
    SourceSpan,
)
from sanad.models.regulatory import (
    ArticleReference,
    RagHealth,
    RegulatoryAnswerResult,
    RegulatoryEvidence,
    RegulatoryEvidenceResult,
    RegulatoryQuery,
    RetrievalConfig,
    SourceDocument,
)

__all__ = [
    "AnalysisBundle",
    "AnalysisFinding",
    "AnalysisStatus",
    "ArticleReference",
    "ContractAnalysisResult",
    "CvAnalysisResult",
    "EvidenceReference",
    "FindingStatus",
    "TargetJob",
    "ComparedContract",
    "ContractComparisonResult",
    "DimensionComparison",
    "Recommendation",
    "DocumentRole",
    "OrchestratorResult",
    "Route",
    "SanadRequest",
    "TaskHint",
    "UploadedDocument",
    "ContractExtraction",
    "CvExtraction",
    "DocumentPage",
    "DocumentSection",
    "DocumentStatus",
    "DocumentTable",
    "ExtractedField",
    "FieldStatus",
    "FileType",
    "ParsedDocument",
    "SourceSpan",
    "ErrorInfo",
    "RagHealth",
    "RegulatoryAnswerResult",
    "RegulatoryEvidence",
    "RegulatoryEvidenceResult",
    "RegulatoryQuery",
    "ResultStatus",
    "RetrievalConfig",
    "SourceDocument",
]
