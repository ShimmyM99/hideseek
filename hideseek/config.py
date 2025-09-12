import yaml
import os
from typing import Dict, Any, List
from pathlib import Path


class HideSeekConfig:
    """Configuration manager for HideSeek camouflage testing system"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'detection.color_similarity_threshold')"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_detection_thresholds(self) -> Dict[str, float]:
        """Get detection threshold parameters"""
        return self.get('detection', {})
    
    def get_standard_distances(self) -> List[int]:
        """Get standard viewing distances in meters"""
        return self.get('distances.standard', [5, 10, 25, 50, 100])
    
    def get_environment_config(self, env_type: str) -> Dict[str, Any]:
        """Get configuration for specific environment type"""
        return self.get(f'environments.{env_type}', {})
    
    def get_scoring_weights(self, environment: str = None) -> Dict[str, float]:
        """Get scoring weights, optionally adjusted for specific environment"""
        base_weights = self.get('scoring.weights', {})
        
        if environment:
            adjustments = self.get(f'scoring.environment_adjustments.{environment}', {})
            # Apply environment-specific adjustments
            weights = base_weights.copy()
            weights.update(adjustments)
            return weights
        
        return base_weights
    
    def get_analysis_params(self) -> Dict[str, Any]:
        """Get analysis parameters"""
        return self.get('analysis', {})
    
    def get_image_settings(self) -> Dict[str, Any]:
        """Get image processing settings"""
        return self.get('image', {})
    
    def get_lighting_config(self) -> Dict[str, Any]:
        """Get lighting simulation parameters"""
        return self.get('lighting', {})
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.get('output', {})
    
    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        required_sections = ['detection', 'distances', 'environments', 'scoring', 'analysis']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return True


# Global configuration instance
config = HideSeekConfig()