# TMG - TissueMultiGraph Analysis and Visualization Package

TMG is a comprehensive Python package for analyzing and visualizing tissue data using multi-layer graph representations. This package provides core datastructure (TMG) and functionality for tissue analysis and visualization.

## Features

### Analysis Module
- **TissueMultiGraph**: Core class for managing multi-layer graph representations of tissues
- **TissueGraph**: Single-layer graph representation of biospatial units
- **Classification**: Advanced classification tools for cell types and regions

### Visualization Module  
- **Spatial Maps**: Comprehensive spatial visualization tools
- **Multi-section Views**: Tools for visualizing multiple tissue sections
- **Interactive Plots**: Scatter plots, histograms, and specialized visualizations
- **Color Management**: Advanced color mapping and legend generation

### Utilities Module
- **Data Processing**: Core data manipulation and processing functions
- **Geometric Operations**: Spatial geometry and polygon operations
- **File I/O**: Robust file input/output utilities
- **Color Management**: Color utilities for visualization

## Installation

### Recommended: Using Conda Environment
```bash
# Create and activate the environment
conda env create -f environment_TMG.yml
conda activate TMG

# Install TMG package in development mode
cd TMG
pip install -e .
```

### Alternative: Using pip only
```bash
cd TMG
pip install -e .
```

### With development dependencies:
```bash
pip install -e .[dev]
```

## Dependencies

### Conda Environment Setup
Create the environment using the provided file:
```bash
conda env create -f environment_TMG.yml
conda activate TMG
```

### Key Dependencies
- **Core Scientific**: NumPy, Pandas, SciPy for data manipulation
- **Visualization**: Matplotlib, Seaborn, colorcet for plotting and visualization
- **Machine Learning**: scikit-learn, PyTorch for analysis algorithms
- **Graph Analysis**: python-igraph, leidenalg, pynndescent for graph operations
- **Single-cell**: AnnData for single-cell data handling
- **Image Processing**: Pillow, OpenCV, scikit-image for image analysis
- **Specialized**: colormath for color space operations, umap-learn for dimensionality reduction

## Quick Start

```python
import TMG

# Create a TissueMultiGraph object
tmg = TMG.TissueMultiGraph(basepath="path/to/data")

# Create cell layer analysis
tmg.create_cell_layer()

# Generate visualizations
view = TMG.SingleMapView(tmg, section="section_name")
view.show()
```

## Package Structure

```
TMG/
├── Analysis/          # Core analysis functionality
│   ├── TissueGraph.py    # Main graph classes
│   └── Classification.py # Classification tools
├── Visualization/     # Visualization tools
│   ├── Viz.py           # Main visualization classes
│   ├── cell_colors.py   # Color management
│   └── utils.py         # Visualization utilities
└── Utils/             # Core utilities
    ├── basicu.py        # Basic utility functions
    ├── geomu.py         # Geometric operations
    ├── fileu.py         # File operations
    └── ...              # Additional utilities
```

