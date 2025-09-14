"""
HideSeek - Camouflage Testing System

A professional Python application for testing and evaluating camouflage effectiveness.
This system quantitatively measures how well camouflaged objects "hide" in their 
environment and how easily they can be "sought" (detected) through computer vision analysis.
"""

__version__ = "1.0.0"
__author__ = "HideSeek Development Team"
__email__ = "99cvteam@gmail.com"

# Lazy imports to avoid dependency issues at startup
def get_image_loader():
    from .core.image_loader import HideSeekImageLoader
    return HideSeekImageLoader

def get_data_manager():
    from .core.data_manager import TestDataManager
    return TestDataManager

def get_report_generator():
    from .core.report_generator import HideSeekReportGenerator
    return HideSeekReportGenerator

__all__ = [
    "get_image_loader",
    "get_data_manager", 
    "get_report_generator"
]