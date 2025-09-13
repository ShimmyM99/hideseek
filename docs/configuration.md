# HideSeek Configuration Guide

## Configuration Files

HideSeek uses YAML configuration files for customizing analysis parameters, scoring weights, and system behavior.

### Default Configuration Location
- **Primary:** `config.yaml` in project root
- **User Config:** `~/.hideseek/config.yaml`
- **System Config:** `/etc/hideseek/config.yaml`

### Configuration Hierarchy
1. Command-line arguments (highest priority)
2. Custom config file (`--config` option)
3. Project config file (`./config.yaml`)
4. User config file (`~/.hideseek/config.yaml`)
5. Default built-in configuration (lowest priority)

## Configuration Sections

### Analysis Configuration
```yaml
analysis:
  # Color analysis settings
  color_blending:
    gamma_correction: true
    white_balance: true
    delta_e_method: "ciede2000"
    spatial_window_size: 32
    
  # Pattern analysis settings
  pattern_disruption:
    feature_detectors: ["orb", "sift", "brisk"]
    texture_methods: ["lbp", "gabor"]
    edge_detection_threshold: 50
    
  # Brightness analysis settings  
  brightness_contrast:
    illumination_conditions: ["daylight", "twilight", "night"]
    local_contrast_window: 16
    atmospheric_model: true
    
  # Distance simulation settings
  distance_detection:
    test_distances: [5, 10, 25, 50, 100, 200]
    atmospheric_visibility: 15000  # meters
    observer_acuity: 1.0  # arcminutes
```

### Scoring Configuration
```yaml
scoring:
  # Component weights (must sum to 1.0)
  weights:
    color_blending: 0.30
    pattern_disruption: 0.25
    brightness_contrast: 0.20
    distance_detection: 0.25
    
  # Environment-specific adjustments
  environment_adjustments:
    woodland:
      color_blending: 0.30
      pattern_disruption: 0.25
      brightness_contrast: 0.20
      distance_detection: 0.25
    desert:
      color_blending: 0.25
      pattern_disruption: 0.20
      brightness_contrast: 0.35
      distance_detection: 0.20
    urban:
      color_blending: 0.20
      pattern_disruption: 0.35
      brightness_contrast: 0.25
      distance_detection: 0.20
      
  # Score thresholds
  thresholds:
    excellent: 90
    good: 75
    acceptable: 60
    poor: 40
```

### Visualization Configuration
```yaml
visualization:
  # Report generation settings
  reports:
    default_format: "pdf"
    include_visualizations: true
    color_space: "srgb"
    dpi: 300
    
  # Chart settings
  charts:
    radar_chart:
      show_grid: true
      show_legend: true
      colors: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    comparison_matrix:
      colormap: "viridis"
      show_values: true
      
  # Image display settings
  display:
    max_width: 1920
    max_height: 1080
    gamma_correction: 2.2
```

### Performance Configuration
```yaml
performance:
  # Processing settings
  processing:
    max_workers: 4
    chunk_size: 1000
    memory_limit: "2GB"
    
  # Caching settings
  caching:
    enabled: true
    max_size: "1GB" 
    ttl: 86400  # 24 hours
    
  # Optimization settings
  optimization:
    use_gpu: false
    precision: "float32"
    parallel_analysis: true
```

### Logging Configuration
```yaml
logging:
  level: "INFO"
  format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
  
  handlers:
    console:
      enabled: true
      level: "INFO"
    file:
      enabled: true
      level: "DEBUG"
      filename: "hideseek.log"
      max_size: "10MB"
      backup_count: 5
```

## Environment-Specific Configurations

### Woodland Environment
```yaml
woodland:
  color_palette: ["#2d4a2b", "#4a5c48", "#6b7c6b", "#8b9d8b"]
  texture_emphasis: true
  shadow_importance: high
  seasonal_variations: true
```

### Desert Environment  
```yaml
desert:
  color_palette: ["#c4965a", "#e4c18a", "#f4e1ba", "#d4b57a"]
  brightness_critical: true
  heat_shimmer_simulation: true
  sand_texture_matching: true
```

### Urban Environment
```yaml
urban:
  geometric_patterns: true
  material_variety: ["concrete", "glass", "metal", "brick"]
  lighting_complexity: high
  vertical_emphasis: true
```

## Custom Algorithm Parameters

### Color Analysis Parameters
```yaml
color_analysis:
  ciede2000:
    kl: 1.0  # Lightness weight
    kc: 1.0  # Chroma weight  
    kh: 1.0  # Hue weight
    
  lab_conversion:
    white_point: "D65"
    observer: "2"
    
  spatial_analysis:
    window_overlap: 0.5
    minimum_window_size: 16
```

### Pattern Analysis Parameters
```yaml
pattern_analysis:
  orb:
    nfeatures: 500
    scaleFactor: 1.2
    nlevels: 8
    
  sift:
    nfeatures: 0
    nOctaveLayers: 3
    contrastThreshold: 0.04
    
  texture_analysis:
    gabor_frequencies: [0.1, 0.3, 0.5]
    gabor_orientations: [0, 45, 90, 135]
    lbp_radius: 3
    lbp_points: 24
```

## CLI Configuration Override

### Command-Line Options
```bash
# Override config file location
hideseek test --config custom_config.yaml --image test.jpg

# Override specific parameters
hideseek test --image test.jpg \
  --scoring-weight-color 0.4 \
  --scoring-weight-pattern 0.3

# Environment-specific overrides
hideseek test --image test.jpg --environment woodland \
  --analysis-gamma-correction false
```

### Environment Variables
```bash
export HIDESEEK_CONFIG="/path/to/config.yaml"
export HIDESEEK_LOG_LEVEL="DEBUG"
export HIDESEEK_CACHE_DIR="/tmp/hideseek_cache"
```

## Advanced Configuration

### Custom Environments
```yaml
custom_environments:
  jungle:
    base_environment: "tropical"
    color_palette: ["#1a2f1a", "#2d4a2d", "#3f5f3f"]
    humidity_effects: true
    dense_vegetation: true
    
  snow_forest:
    base_environment: "arctic"
    seasonal_variant: "winter"
    snow_coverage: 0.8
    bare_tree_emphasis: true
```

### Plugin Configuration
```yaml
plugins:
  enabled: ["thermal_analysis", "multispectral_analysis"]
  
  thermal_analysis:
    temperature_range: [-20, 50]  # Celsius
    emissivity_correction: true
    
  multispectral_analysis:
    bands: ["red", "green", "blue", "nir"]
    vegetation_indices: ["ndvi", "savi"]
```

## Validation and Testing

### Configuration Validation
```bash
# Validate configuration file
hideseek config --validate config.yaml

# Show effective configuration
hideseek config --show

# Test configuration with sample
hideseek config --test --image sample.jpg
```

### Configuration Templates
HideSeek provides templates for common use cases:
- `templates/military.yaml`: Military assessment configuration
- `templates/wildlife.yaml`: Wildlife photography configuration  
- `templates/research.yaml`: Academic research configuration
- `templates/industrial.yaml`: Industrial camouflage configuration