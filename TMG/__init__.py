"""
TMG - TissueMultiGraph Analysis and Visualization Package

This package provides functionality for analyzing and visualizing tissue data
using multi-layer graph representations. It includes tools for:

- Analysis: TissueGraph analysis, classification, and spatial analysis
- Visualization: Comprehensive visualization tools for tissue data
- Utilities: Core utility functions for data processing and manipulation

Main modules:
- TMG.Analysis: Core analysis functionality including TissueGraph and Classification
- TMG.Visualization: Visualization tools and plotting functionality  
- TMG.Utils: Utility functions for data processing
"""

__version__ = "1.0.0"

# Import main submodules
from . import Analysis
from . import Visualization
from . import Utils

__all__ = [
    'Analysis',
    'Visualization', 
    'Utils'
]
