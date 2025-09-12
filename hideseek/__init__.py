"""
HideSeek - Camouflage Testing System

A professional Python application for testing and evaluating camouflage effectiveness.
This system quantitatively measures how well camouflaged objects "hide" in their 
environment and how easily they can be "sought" (detected) through computer vision analysis.
"""

__version__ = "1.0.0"
__author__ = "HideSeek Development Team"
__email__ = "hideseek@example.com"

from .core.image_loader import HideSeekImageLoader
from .core.data_manager import TestDataManager
from .core.report_generator import HideSeekReportGenerator

__all__ = [
    "HideSeekImageLoader",
    "TestDataManager", 
    "HideSeekReportGenerator"
]