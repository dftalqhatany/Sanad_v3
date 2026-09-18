"""Structured field extraction from ParsedDocument (contracts and CVs).

Depends only on sanad.models: it neither parses files itself nor imports the regulatory RAG.
"""

from sanad.extraction.contract import ContractExtractor, extract_contract
from sanad.extraction.cv import CvExtractor, extract_cv

__all__ = ["ContractExtractor", "CvExtractor", "extract_contract", "extract_cv"]
