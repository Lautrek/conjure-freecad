"""
Shared utilities for Conjure clients.

This module provides common functionality used across all client adapters
(Blender, FreeCAD, Fusion 360, etc.) to ensure unified behavior.
"""

from .feedback import (
    CommandFeedback,
    FeedbackClient,
    get_feedback_client,
    init_feedback_client,
    stop_feedback_client,
)
from .materials import EngineeringMaterial, MaterialCache, MaterialsClient
from .standards import GearSpec, LocalStandardsLibrary, get_standards_library

__all__ = [
    "MaterialsClient",
    "EngineeringMaterial",
    "MaterialCache",
    "GearSpec",
    "LocalStandardsLibrary",
    "get_standards_library",
    # Feedback client
    "FeedbackClient",
    "CommandFeedback",
    "get_feedback_client",
    "init_feedback_client",
    "stop_feedback_client",
]
