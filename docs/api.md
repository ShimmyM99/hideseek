# HideSeek API Documentation

## Core Classes

### HideSeekImageLoader
```python
from hideseek import get_image_loader

loader = get_image_loader()
image = loader.load_test_image("camouflage.jpg")
```

**Methods:**
- `load_test_image(path)`: Load single image with preprocessing
- `load_batch_images(directory)`: Load multiple images
- `validate_image_format(path)`: Check image format compatibility

### TestDataManager
```python
from hideseek import get_data_manager

manager = get_data_manager()
session = manager.create_session("Test Session")
```

**Methods:**
- `create_session(name, description)`: Create analysis session
- `save_test_results(session_id, results)`: Save analysis results
- `get_session_history()`: Retrieve past sessions

### HideSeekReportGenerator
```python
from hideseek import get_report_generator

generator = get_report_generator()
generator.generate_comprehensive_report(results, "report.pdf")
```

**Methods:**
- `generate_comprehensive_report(results, output_path)`: Full PDF report
- `generate_quick_report(results)`: Simple text summary
- `create_comparison_report(multiple_results)`: Compare multiple analyses

## Analysis Pipeline

### PipelineController
```python
from hideseek.analysis.pipeline_controller import PipelineController
from hideseek import get_data_manager

controller = PipelineController(get_data_manager())
results = controller.execute_full_analysis(image, background)
```

**Analysis Modes:**
- `execute_quick_analysis()`: Fast, basic analysis
- `execute_full_analysis()`: Complete analysis suite
- `execute_detailed_analysis()`: Extended analysis with visualizations

### Individual Analyzers

#### ColorBlendingAnalyzer
```python
from hideseek.analysis.color_analyzer import ColorBlendingAnalyzer

analyzer = ColorBlendingAnalyzer()
score = analyzer.analyze_color_blending(image, background)
```

#### PatternDisruptionAnalyzer
```python
from hideseek.analysis.pattern_analyzer import PatternDisruptionAnalyzer

analyzer = PatternDisruptionAnalyzer()
score = analyzer.analyze_pattern_disruption(image, background)
```

## Scoring System

### HideSeekScoringEngine
```python
from hideseek.scoring.scoring_engine import HideSeekScoringEngine

engine = HideSeekScoringEngine()
final_score = engine.calculate_weighted_score(analysis_results)
```

**Key Methods:**
- `calculate_weighted_score()`: Compute final weighted score
- `generate_detailed_breakdown()`: Score component analysis
- `create_radar_chart()`: Visual score representation

## Configuration

### Custom Configuration
```python
from hideseek.config import HideSeekConfig

config = HideSeekConfig("custom_config.yaml")
```

**Configuration Sections:**
- `analysis`: Analysis pipeline settings
- `scoring`: Score weighting parameters
- `visualization`: Report generation options
- `performance`: Processing optimization settings

## Error Handling

All HideSeek functions raise appropriate exceptions:
- `ImageLoadError`: Image loading/format issues
- `AnalysisError`: Analysis pipeline failures  
- `ConfigurationError`: Configuration problems
- `InsufficientDataError`: Missing required inputs

## Examples

### Basic Usage
```python
from hideseek import get_image_loader
from hideseek.analysis.pipeline_controller import PipelineController
from hideseek import get_data_manager

# Load images
loader = get_image_loader()
camo_image = loader.load_test_image("camouflage.jpg")
bg_image = loader.load_test_image("background.jpg")

# Run analysis  
controller = PipelineController(get_data_manager())
results = controller.execute_full_analysis(camo_image, bg_image)

print(f"Camouflage effectiveness: {results['overall_score']}/100")
```

### Batch Processing
```python
import os
from pathlib import Path

# Process directory of images
image_dir = Path("./test_images")
results = []

for image_path in image_dir.glob("*.jpg"):
    image = loader.load_test_image(str(image_path))
    result = controller.execute_quick_analysis(image)
    results.append((image_path.name, result))

# Generate comparison report
generator = get_report_generator()
generator.create_comparison_report(results, "batch_report.pdf")
```