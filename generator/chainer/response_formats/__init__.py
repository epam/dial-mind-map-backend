"""
Exposes key Pydantic models from the response_models submodule for easy
access, defining the public API of the 'models' package.

This flattens the namespace, so users can import models like:
`from generator.models import KnowledgeFragment`
instead of the more verbose:
`from generator.models.response_models import KnowledgeFragment`
"""

from typing import Any, Protocol, TypedDict

from .response_models import AnswerPart, Results

# Define the public API of this package. This controls the behavior of
# `from <package> import *` and is used by static analysis tools.
__all__ = [
    "ClusterSynthesisResult",
    "DeduplicationResult",
    "ExtractionResult",
    "KnowledgeFragment",
    "PrettyConcepts",
    "RefinedFilterResult",
    "RefinedStyleResult",
    "RenamingResult",
    "ResultRoot",
    "RootClusterSynthesisResult",
    "RootDeduplicationResult",
    "SynthesizedAnswerPart",
    "ValidationResult",
    "RootClusterSynthesisResultProtocol",
    "RefinedStyleResultProtocol",
]

# Re-export models from their container classes to provide a simpler,
# flatter namespace for package users.
ClusterSynthesisResult = Results.ClusterSynthesisResult
DeduplicationResult = Results.DeduplicationResult
ExtractionResult = Results.ExtractionResult
KnowledgeFragment = Results.KnowledgeFragment
PrettyConcepts = Results.PrettyConcepts
RefinedFilterResult = Results.RefinedFilterResult
RefinedStyleResult = Results.RefinedStyleResult
RenamingResult = Results.RenamingResult
ResultRoot = Results.ResultRoot
RootClusterSynthesisResult = Results.RootClusterSynthesisResult
RootDeduplicationResult = Results.RootDeduplicationResult
SynthesizedAnswerPart = AnswerPart.Synthesized
ValidationResult = Results.ValidationResult


class RootClusterSynthesisResultProtocol(Protocol):
    """A protocol describing the shape of the synthesis result."""

    source_ids: list[str]
    synthesis: str


class ValidationDict(TypedDict):
    is_safe: bool
    reason: str


class RefinedStyleResultProtocol(Protocol):
    """A protocol describing the shape of a refined style result."""

    validation: ValidationDict
    language: str
    persona: str
    tone: str
    other_instructions: str

    def model_dump(self) -> dict[str, Any]:
        # The '...' indicates that the body is not implemented here.
        # We just care that the method exists.
        ...
