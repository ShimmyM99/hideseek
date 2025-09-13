# HideSeek - Camouflage Testing System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional Python application for testing and evaluating camouflage effectiveness. HideSeek quantitatively measures how well camouflaged objects "hide" in their environment and how easily they can be "sought" (detected) through computer vision analysis.

## 🎯 Project Purpose

HideSeek analyzes camouflage effectiveness by measuring:
- **Color blending** between hidden objects and backgrounds
- **Pattern disruption** that prevents object recognition  
- **Brightness/contrast matching** across lighting conditions
- **Detection difficulty** at various viewing distances
- **Overall "hideability" score** for camouflage patterns

## 🚀 Features

- **Multi-Modal Analysis**: Color, pattern, brightness, and distance-based detection
- **Scientific Accuracy**: Uses perceptual color spaces (LAB) and industry-standard algorithms
- **Environment Simulation**: Tests across woodland, desert, urban, arctic, and tropical environments
- **Distance Modeling**: Simulates detection probability at standard viewing distances (5m-100m)
- **Professional Reports**: Generates comprehensive PDF/HTML reports with visualizations
- **CLI Interface**: Easy-to-use command-line interface for batch processing

## 📋 Requirements

- Python 3.10+
- OpenCV 4.8+
- NumPy, SciPy, scikit-image
- matplotlib, seaborn, plotly
- colour-science for accurate color analysis

## 🛠️ Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/ShimmyM99/hideseek.git
cd hideseek

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install HideSeek in development mode
pip install -e .
```

### Included Test Images
The project includes sample camouflage images for testing:
- `./data/test_images/good/` - Examples of effective camouflage patterns
- `./data/test_images/bad/` - Examples of poor camouflage for comparison

### Using pip (when available on PyPI)

```bash
pip install hideseek
```

## 🎮 Quick Start

### CLI Usage

```bash
# Quick analysis (fastest) - using included test images
python -m hideseek quick --image "./data/test_images/good/amazing-wild-animal-camouflage-nature-8-59258edad4f22__700.jpg"

# Full analysis with background reference
python -m hideseek test --image "./data/test_images/good/8nksrdfqcxz91.jpg" --background "./data/test_images/good/images.jpeg" --output report.pdf

# Detailed analysis with visualizations
python -m hideseek detailed --image "./data/test_images/good/images2.jpeg" --environment woodland --visualizations

# Batch processing all good camouflage examples
python -m hideseek batch --directory "./data/test_images/good" --environment woodland --format json

# Compare good vs bad camouflage
python -m hideseek compare --patterns "./data/test_images/good/amazing-wild-animal-camouflage-nature-8-59258edad4f22__700.jpg" "./data/test_images/bad/original.jpg" --output comparison.pdf

# System information
python -m hideseek info
```

### Python API

```python
from hideseek import get_image_loader
from hideseek.analysis.pipeline_controller import PipelineController
from hideseek import get_data_manager

# Load test images (using included samples)
loader = get_image_loader()
camo_img = loader.load_test_image("./data/test_images/good/amazing-wild-animal-camouflage-nature-8-59258edad4f22__700.jpg")
bg_img = loader.load_test_image("./data/test_images/good/images.jpeg")

# Run analysis
controller = PipelineController(get_data_manager())
results = controller.execute_quick_analysis(camo_img)
print(f"Camouflage effectiveness: {results['overall_score']}/100")
```

## 📊 Analysis Modules

### Color Blending Analysis
- **Gamma linearization** for accurate color representation
- **White balance correction** for consistent lighting
- **CIEDE2000 color difference** calculations
- **Perceptual color matching** in LAB color space

### Pattern Disruption Analysis  
- **Multi-feature detection** (ORB, SIFT, SURF)
- **Edge continuity measurement**
- **Texture similarity analysis** using Gabor filters
- **Fractal dimension** for pattern complexity
- **Shape breakup detection**

### Brightness & Contrast Matching
- **Luminance extraction** for brightness analysis
- **Local contrast mapping**
- **Shadow pattern analysis**
- **Multi-illumination testing** (daylight, twilight, night, IR)

### Distance-Based Detection
- **Angular size calculations**
- **Atmospheric blur simulation**
- **Visual acuity modeling**
- **Detection probability curves**
- **Critical detection distance** estimation

## 🏗️ Project Architecture

```
hideseek/
├── hideseek/
│   ├── core/           # I/O, data management, reporting
│   ├── analysis/       # Analysis pipelines
│   ├── scoring/        # Scoring algorithms
│   ├── visualization/  # Report generation
│   └── utils/          # Utilities and helpers
├── tests/              # Comprehensive test suite
├── data/               # Environment templates and test data
├── docs/               # Documentation
└── scripts/            # Setup and utility scripts
```

## 📈 Scoring Methodology

HideSeek uses a scientifically-grounded scoring system:

- **Color Blending (30%)**: CIEDE2000 color difference analysis
- **Pattern Disruption (25%)**: Feature matching and texture analysis  
- **Brightness Matching (20%)**: Luminance and contrast analysis
- **Distance Effectiveness (25%)**: Detection probability modeling

Scores are weighted and can be adjusted for specific environments:
- Desert: Emphasizes brightness matching
- Arctic: Critical brightness analysis for snow conditions  
- Urban: Enhanced pattern disruption weighting

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hideseek

# Run specific test modules
pytest tests/test_color_analyzer.py
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Analysis Methodology](docs/methodology.md)  
- [Tutorial Notebook](docs/examples.ipynb)
- [Configuration Guide](docs/configuration.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Scientific validation methods

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Development Status

### ✅ Phase 1: Foundation (Complete)
- [x] Core I/O and configuration system
- [x] Project structure and Git setup
- [x] Image loading and preprocessing
- [x] Data management and session handling
- [x] Basic report generation

### ✅ Phase 2: Core Analysis (Complete)
- [x] Color blending analysis pipeline with CIEDE2000
- [x] Pattern disruption analysis with multi-feature detection
- [x] Brightness and contrast analysis with multi-illumination
- [x] Distance-based detection simulation
- [x] Environmental context analyzer

### ✅ Phase 3: Integration (Complete)
- [x] Analysis router with intelligent pipeline selection
- [x] Pipeline controller with multiple analysis modes
- [x] Comprehensive scoring engine with adaptive weighting
- [x] Professional CLI interface with 7 operation modes
- [x] Multi-format reporting (PDF, HTML, JSON, CSV)

### 🚀 Phase 4: Future Enhancements
- [ ] Machine learning integration for pattern recognition
- [ ] Real-time analysis capabilities
- [ ] Web interface and REST API  
- [ ] Thermal and multispectral analysis
- [ ] Advanced environmental simulation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ShimmyM99/hideseek/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ShimmyM99/hideseek/discussions)  
- **Repository**: [HideSeek on GitHub](https://github.com/ShimmyM99/hideseek)

## 🔬 Scientific Applications

HideSeek is designed for:
- **Military camouflage evaluation**
- **Wildlife photography gear testing**
- **Hunting equipment assessment** 
- **Academic research** in computer vision and camouflage
- **Industrial camouflage design**

---

**HideSeek** - Making the invisible, measurable.