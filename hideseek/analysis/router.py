import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from scipy import stats

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('analysis_router')


class HideSeekAnalysisRouter:
    """
    Intelligent analysis router that determines appropriate analysis pipelines 
    based on image characteristics and environment detection.
    """
    
    def __init__(self):
        self.analysis_params = config.get_analysis_params()
        self.environment_configs = config.get('environments', {})
        
        # Initialize environment classifier features
        self._init_environment_features()
        
        logger.info("AnalysisRouter initialized with intelligent pipeline selection")
    
    def _init_environment_features(self):
        """Initialize features used for environment classification"""
        self.environment_features = {
            'woodland': {
                'dominant_hue_ranges': [(35, 85), (15, 35)],  # Green and brown ranges
                'texture_complexity_threshold': 0.6,
                'brightness_range': (40, 180),
                'saturation_preference': 'medium'
            },
            'desert': {
                'dominant_hue_ranges': [(10, 30)],  # Sandy/brown range
                'texture_complexity_threshold': 0.3,
                'brightness_range': (120, 255),
                'saturation_preference': 'low'
            },
            'urban': {
                'dominant_hue_ranges': [(0, 180)],  # All hues (neutral colors)
                'texture_complexity_threshold': 0.8,
                'brightness_range': (20, 200),
                'saturation_preference': 'low'
            },
            'arctic': {
                'dominant_hue_ranges': [(90, 120), (0, 20)],  # Blue-white range
                'texture_complexity_threshold': 0.2,
                'brightness_range': (180, 255),
                'saturation_preference': 'very_low'
            },
            'tropical': {
                'dominant_hue_ranges': [(45, 85)],  # Intense green range
                'texture_complexity_threshold': 0.9,
                'brightness_range': (60, 220),
                'saturation_preference': 'high'
            }
        }
    
    def analyze_image_characteristics(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze fundamental image characteristics for pipeline selection.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Dictionary containing image characteristics
        """
        logger.debug("Analyzing image characteristics for pipeline routing")
        
        characteristics = {}
        
        # Basic image properties
        height, width = img.shape[:2]
        characteristics['dimensions'] = (width, height)
        characteristics['aspect_ratio'] = width / height
        characteristics['total_pixels'] = width * height
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Color characteristics
        characteristics['color_properties'] = self._analyze_color_properties(img, hsv, lab)
        
        # Texture and pattern characteristics  
        characteristics['texture_properties'] = self._analyze_texture_properties(gray)
        
        # Brightness and contrast characteristics
        characteristics['brightness_properties'] = self._analyze_brightness_properties(lab)
        
        # Edge and shape characteristics
        characteristics['edge_properties'] = self._analyze_edge_properties(gray)
        
        # Overall complexity score
        characteristics['complexity_score'] = self._calculate_complexity_score(characteristics)
        
        logger.debug(f"Image characteristics: complexity={characteristics['complexity_score']:.2f}")
        
        return characteristics
    
    def _analyze_color_properties(self, bgr_img: np.ndarray, hsv_img: np.ndarray, 
                                 lab_img: np.ndarray) -> Dict[str, Any]:
        """Analyze color-related properties"""
        h, s, v = cv2.split(hsv_img)
        l, a, b = cv2.split(lab_img)
        
        return {
            'dominant_hues': self._get_dominant_hues(h),
            'saturation_stats': {
                'mean': np.mean(s),
                'std': np.std(s),
                'range': (np.min(s), np.max(s))
            },
            'value_stats': {
                'mean': np.mean(v),
                'std': np.std(v),
                'range': (np.min(v), np.max(v))
            },
            'lab_stats': {
                'l_mean': np.mean(l),
                'a_mean': np.mean(a),
                'b_mean': np.mean(b),
                'l_std': np.std(l),
                'chroma': np.sqrt(np.mean(a**2 + b**2))
            },
            'color_diversity': self._calculate_color_diversity(bgr_img)
        }
    
    def _analyze_texture_properties(self, gray_img: np.ndarray) -> Dict[str, Any]:
        """Analyze texture and pattern properties"""
        # Calculate texture using local binary patterns approximation
        texture_energy = self._calculate_texture_energy(gray_img)
        
        # Calculate directional gradients
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        return {
            'texture_energy': texture_energy,
            'gradient_stats': {
                'mean_magnitude': np.mean(gradient_magnitude),
                'std_magnitude': np.std(gradient_magnitude),
                'direction_consistency': self._calculate_direction_consistency(gradient_direction)
            },
            'pattern_regularity': self._assess_pattern_regularity(gray_img),
            'fractal_dimension': self._estimate_fractal_dimension(gray_img)
        }
    
    def _analyze_brightness_properties(self, lab_img: np.ndarray) -> Dict[str, Any]:
        """Analyze brightness and luminance properties"""
        l_channel = lab_img[:, :, 0]
        
        # Calculate local contrast
        local_contrast = self._calculate_local_contrast(l_channel)
        
        # Histogram analysis
        hist, _ = np.histogram(l_channel, bins=256, range=(0, 255))
        hist_normalized = hist / np.sum(hist)
        
        return {
            'luminance_stats': {
                'mean': np.mean(l_channel),
                'std': np.std(l_channel),
                'range': (np.min(l_channel), np.max(l_channel)),
                'median': np.median(l_channel)
            },
            'contrast_measures': {
                'rms_contrast': np.std(l_channel) / np.mean(l_channel) if np.mean(l_channel) > 0 else 0,
                'michelson_contrast': self._calculate_michelson_contrast(l_channel),
                'local_contrast_mean': np.mean(local_contrast),
                'local_contrast_std': np.std(local_contrast)
            },
            'histogram_properties': {
                'entropy': -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7)),
                'skewness': stats.skew(l_channel.flatten()),
                'kurtosis': stats.kurtosis(l_channel.flatten())
            }
        }
    
    def _analyze_edge_properties(self, gray_img: np.ndarray) -> Dict[str, Any]:
        """Analyze edge and contour properties"""
        # Canny edge detection
        edges = cv2.Canny(gray_img, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'edge_density': np.sum(edges > 0) / edges.size,
            'contour_count': len(contours),
            'average_contour_length': np.mean([cv2.arcLength(contour, False) for contour in contours]) if contours else 0,
            'contour_complexity': self._calculate_contour_complexity(contours)
        }
    
    def determine_environment_type(self, img: np.ndarray) -> str:
        """
        Determine the most likely environment type based on image analysis.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Environment type string
        """
        characteristics = self.analyze_image_characteristics(img)
        
        environment_scores = {}
        
        for env_type, features in self.environment_features.items():
            score = self._calculate_environment_match_score(characteristics, features)
            environment_scores[env_type] = score
        
        # Find best matching environment
        best_env = max(environment_scores.items(), key=lambda x: x[1])
        confidence = best_env[1]
        
        logger.info(f"Environment detected: {best_env[0]} (confidence: {confidence:.2f})")
        
        # Log all scores for debugging
        for env, score in sorted(environment_scores.items(), key=lambda x: x[1], reverse=True):
            logger.debug(f"  {env}: {score:.2f}")
        
        return best_env[0]
    
    def select_analysis_pipelines(self, img_characteristics: Dict[str, Any]) -> List[str]:
        """
        Select appropriate analysis pipelines based on image characteristics.
        
        Args:
            img_characteristics: Results from analyze_image_characteristics
            
        Returns:
            List of pipeline names to execute
        """
        selected_pipelines = []
        
        # Always include core pipelines
        core_pipelines = ['color_blending', 'brightness_matching']
        selected_pipelines.extend(core_pipelines)
        
        # Add pattern analysis if image has sufficient texture complexity
        texture_props = img_characteristics.get('texture_properties', {})
        if texture_props.get('texture_energy', 0) > 0.3:
            selected_pipelines.append('pattern_disruption')
        
        # Add distance analysis if image resolution is sufficient
        dimensions = img_characteristics.get('dimensions', (0, 0))
        if min(dimensions) >= 200:  # Minimum resolution for distance analysis
            selected_pipelines.append('distance_detection')
        
        # Add environmental analysis if complexity is high
        complexity = img_characteristics.get('complexity_score', 0)
        if complexity > 0.5:
            selected_pipelines.append('environmental_context')
        
        logger.info(f"Selected pipelines: {selected_pipelines}")
        return selected_pipelines
    
    def route_to_pipeline(self, img: np.ndarray, pipeline: str, 
                         background_img: np.ndarray = None, 
                         options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route analysis to specific pipeline.
        
        Args:
            img: Input camouflage image
            pipeline: Pipeline name to execute
            background_img: Optional background reference image
            options: Pipeline-specific options
            
        Returns:
            Analysis results from the specified pipeline
        """
        logger.debug(f"Routing to pipeline: {pipeline}")
        
        if options is None:
            options = {}
        
        try:
            if pipeline == 'color_blending':
                from .color_analyzer import ColorBlendingAnalyzer
                analyzer = ColorBlendingAnalyzer()
                return analyzer.analyze_color_blending(img, background_img, options)
                
            elif pipeline == 'pattern_disruption':
                from .pattern_analyzer import PatternDisruptionAnalyzer
                analyzer = PatternDisruptionAnalyzer()
                return analyzer.analyze_pattern_disruption(img, background_img, options)
                
            elif pipeline == 'brightness_matching':
                from .brightness_analyzer import BrightnessContrastAnalyzer
                analyzer = BrightnessContrastAnalyzer()
                return analyzer.analyze_brightness_contrast(img, background_img, options)
                
            elif pipeline == 'distance_detection':
                from .distance_simulator import DistanceDetectionSimulator
                analyzer = DistanceDetectionSimulator()
                return analyzer.simulate_distance_detection(img, options)
                
            elif pipeline == 'environmental_context':
                from .environment_analyzer import EnvironmentalContextAnalyzer
                analyzer = EnvironmentalContextAnalyzer()
                return analyzer.analyze_environmental_context(img, options)
            
            else:
                raise ValueError(f"Unknown pipeline: {pipeline}")
                
        except ImportError as e:
            logger.warning(f"Pipeline {pipeline} not available: {str(e)}")
            return {'error': f'Pipeline not implemented: {pipeline}', 'score': 0.0}
        except Exception as e:
            logger.error(f"Error in pipeline {pipeline}: {str(e)}")
            return {'error': str(e), 'score': 0.0}
    
    def aggregate_pipeline_results(self, results: List[Dict[str, Any]], 
                                  environment_type: str = None) -> Dict[str, Any]:
        """
        Aggregate results from multiple analysis pipelines.
        
        Args:
            results: List of pipeline results
            environment_type: Environment type for weighted scoring
            
        Returns:
            Aggregated analysis results
        """
        logger.debug("Aggregating pipeline results")
        
        aggregated = {
            'component_scores': {},
            'detailed_results': {},
            'pipeline_errors': [],
            'analysis_metadata': {
                'pipelines_executed': [],
                'environment_type': environment_type
            }
        }
        
        # Process each pipeline result
        for result in results:
            pipeline_name = result.get('pipeline_name', 'unknown')
            aggregated['analysis_metadata']['pipelines_executed'].append(pipeline_name)
            
            # Handle errors
            if 'error' in result:
                aggregated['pipeline_errors'].append({
                    'pipeline': pipeline_name,
                    'error': result['error']
                })
                continue
            
            # Extract component score
            score = result.get('score', 0.0)
            if pipeline_name in ['color_blending', 'pattern_disruption', 
                               'brightness_matching', 'distance_detection']:
                component_name = pipeline_name.replace('_analyzer', '').replace('_detection', '').replace('_matching', '').replace('_blending', '').replace('_disruption', '')
                component_name = component_name.replace('distance', 'distance').replace('brightness', 'brightness').replace('pattern', 'pattern').replace('color', 'color')
                aggregated['component_scores'][component_name] = score
            
            # Store detailed results
            aggregated['detailed_results'][pipeline_name] = result
        
        # Calculate overall score
        aggregated['overall_score'] = self._calculate_overall_score(
            aggregated['component_scores'], environment_type
        )
        
        # Generate recommendations
        aggregated['recommendations'] = self._generate_recommendations(aggregated)
        
        # Add summary statistics
        aggregated['summary'] = self._create_analysis_summary(aggregated)
        
        logger.info(f"Analysis complete. Overall score: {aggregated['overall_score']:.1f}/100")
        
        return aggregated
    
    # Helper methods
    def _get_dominant_hues(self, h_channel: np.ndarray, n_clusters: int = 5) -> List[int]:
        """Extract dominant hues using clustering"""
        h_flat = h_channel.flatten()
        h_flat = h_flat[h_flat > 0]  # Remove black pixels
        
        if len(h_flat) < n_clusters:
            return []
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(h_flat.reshape(-1, 1))
        
        return sorted([int(center[0]) for center in kmeans.cluster_centers_])
    
    def _calculate_color_diversity(self, img: np.ndarray, bins: int = 64) -> float:
        """Calculate color diversity using histogram entropy"""
        # Convert to RGB and calculate 3D histogram
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([rgb], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        hist_normalized = hist / np.sum(hist)
        hist_flat = hist_normalized.flatten()
        
        # Calculate entropy
        entropy = -np.sum(hist_flat * np.log2(hist_flat + 1e-7))
        max_entropy = np.log2(bins**3)
        
        return entropy / max_entropy
    
    def _calculate_texture_energy(self, gray_img: np.ndarray) -> float:
        """Calculate texture energy using gray-level co-occurrence approximation"""
        # Use gradient-based texture measure
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        energy = np.mean(gradient_magnitude) / 255.0  # Normalize
        
        return min(energy, 1.0)
    
    def _calculate_direction_consistency(self, gradient_direction: np.ndarray) -> float:
        """Calculate consistency of gradient directions"""
        # Calculate circular variance of directions
        cos_dirs = np.cos(2 * gradient_direction)
        sin_dirs = np.sin(2 * gradient_direction)
        
        mean_cos = np.mean(cos_dirs)
        mean_sin = np.mean(sin_dirs)
        
        circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
        return 1 - circular_variance  # Convert to consistency measure
    
    def _assess_pattern_regularity(self, gray_img: np.ndarray) -> float:
        """Assess regularity of patterns in the image"""
        # Use autocorrelation to detect regular patterns
        f_transform = np.fft.fft2(gray_img)
        power_spectrum = np.abs(f_transform)**2
        
        # Calculate energy concentration in frequency domain
        total_energy = np.sum(power_spectrum)
        center_energy = np.sum(power_spectrum[gray_img.shape[0]//4:3*gray_img.shape[0]//4,
                                            gray_img.shape[1]//4:3*gray_img.shape[1]//4])
        
        return center_energy / total_energy if total_energy > 0 else 0
    
    def _estimate_fractal_dimension(self, gray_img: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method (simplified)"""
        # Threshold image
        threshold = np.mean(gray_img)
        binary = (gray_img > threshold).astype(int)
        
        # Count boxes at different scales (simplified version)
        scales = [1, 2, 4, 8]
        counts = []
        
        for scale in scales:
            # Coarsen image
            h, w = binary.shape
            coarse_h, coarse_w = h // scale, w // scale
            
            if coarse_h < 1 or coarse_w < 1:
                continue
                
            coarsened = binary[:coarse_h*scale, :coarse_w*scale].reshape(
                coarse_h, scale, coarse_w, scale
            )
            coarsened = np.sum(coarsened, axis=(1, 3)) > 0
            counts.append(np.sum(coarsened))
        
        if len(counts) < 2:
            return 1.5  # Default fractal dimension
        
        # Fit line to log-log plot
        log_scales = np.log([1/s for s in scales[:len(counts)]])
        log_counts = np.log([c + 1 for c in counts])  # +1 to avoid log(0)
        
        if len(log_scales) > 1:
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            return max(1.0, min(2.0, abs(slope)))  # Clamp to reasonable range
        
        return 1.5
    
    def _calculate_local_contrast(self, luminance: np.ndarray, window_size: int = 9) -> np.ndarray:
        """Calculate local contrast using sliding window"""
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
        local_mean = cv2.filter2D(luminance.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((luminance.astype(np.float32) - local_mean)**2, -1, kernel)
        local_std = np.sqrt(local_variance)
        
        # Avoid division by zero
        local_contrast = local_std / (local_mean + 1)
        return local_contrast
    
    def _calculate_michelson_contrast(self, luminance: np.ndarray) -> float:
        """Calculate Michelson contrast"""
        l_max = np.max(luminance)
        l_min = np.min(luminance)
        
        if l_max + l_min == 0:
            return 0.0
        
        return (l_max - l_min) / (l_max + l_min)
    
    def _calculate_contour_complexity(self, contours: List[np.ndarray]) -> float:
        """Calculate average complexity of contours"""
        if not contours:
            return 0.0
        
        complexities = []
        for contour in contours:
            if len(contour) > 5:  # Need at least 5 points
                # Calculate ratio of perimeter to area
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if area > 0:
                    complexity = perimeter**2 / area  # Isoperimetric ratio
                    complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.0
    
    def _calculate_complexity_score(self, characteristics: Dict[str, Any]) -> float:
        """Calculate overall image complexity score"""
        # Weight different complexity factors
        weights = {
            'color_diversity': 0.2,
            'texture_energy': 0.3,
            'edge_density': 0.2,
            'pattern_regularity': 0.15,
            'brightness_entropy': 0.15
        }
        
        scores = {}
        
        # Extract normalized complexity measures
        color_props = characteristics.get('color_properties', {})
        scores['color_diversity'] = color_props.get('color_diversity', 0)
        
        texture_props = characteristics.get('texture_properties', {})
        scores['texture_energy'] = texture_props.get('texture_energy', 0)
        scores['pattern_regularity'] = texture_props.get('pattern_regularity', 0)
        
        edge_props = characteristics.get('edge_properties', {})
        scores['edge_density'] = min(edge_props.get('edge_density', 0) * 10, 1.0)  # Scale edge density
        
        brightness_props = characteristics.get('brightness_properties', {})
        hist_props = brightness_props.get('histogram_properties', {})
        scores['brightness_entropy'] = hist_props.get('entropy', 0) / 8.0  # Normalize entropy
        
        # Calculate weighted average
        complexity = sum(weights.get(key, 0) * scores.get(key, 0) for key in weights.keys())
        
        return max(0.0, min(1.0, complexity))
    
    def _calculate_environment_match_score(self, characteristics: Dict[str, Any], 
                                         env_features: Dict[str, Any]) -> float:
        """Calculate match score for a specific environment"""
        score = 0.0
        
        # Check dominant hues
        color_props = characteristics.get('color_properties', {})
        dominant_hues = color_props.get('dominant_hues', [])
        
        hue_score = 0.0
        for hue_range in env_features['dominant_hue_ranges']:
            for hue in dominant_hues:
                if hue_range[0] <= hue <= hue_range[1]:
                    hue_score += 1.0
        
        if dominant_hues:
            hue_score = min(1.0, hue_score / len(dominant_hues))
        
        score += hue_score * 0.4
        
        # Check texture complexity
        texture_props = characteristics.get('texture_properties', {})
        texture_energy = texture_props.get('texture_energy', 0)
        texture_threshold = env_features['texture_complexity_threshold']
        
        texture_score = 1.0 - abs(texture_energy - texture_threshold)
        score += max(0.0, texture_score) * 0.3
        
        # Check brightness range
        brightness_props = characteristics.get('brightness_properties', {})
        luminance_stats = brightness_props.get('luminance_stats', {})
        mean_brightness = luminance_stats.get('mean', 128)
        
        brightness_range = env_features['brightness_range']
        if brightness_range[0] <= mean_brightness <= brightness_range[1]:
            brightness_score = 1.0
        else:
            # Penalize deviation from range
            if mean_brightness < brightness_range[0]:
                brightness_score = 1.0 - (brightness_range[0] - mean_brightness) / brightness_range[0]
            else:
                brightness_score = 1.0 - (mean_brightness - brightness_range[1]) / (255 - brightness_range[1])
            brightness_score = max(0.0, brightness_score)
        
        score += brightness_score * 0.3
        
        return score
    
    def _calculate_overall_score(self, component_scores: Dict[str, float], 
                               environment_type: str = None) -> float:
        """Calculate weighted overall score"""
        if not component_scores:
            return 0.0
        
        # Get scoring weights (potentially environment-adjusted)
        weights = config.get_scoring_weights(environment_type)
        
        # Map component names to weight keys
        score_mapping = {
            'color': 'color_blending',
            'pattern': 'pattern_disruption', 
            'brightness': 'brightness_matching',
            'distance': 'distance_effectiveness'
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            weight_key = score_mapping.get(component, component)
            weight = weights.get(weight_key, 0.25)  # Default weight
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, aggregated_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on analysis results"""
        recommendations = []
        component_scores = aggregated_results.get('component_scores', {})
        
        # Color-based recommendations
        color_score = component_scores.get('color', 0)
        if color_score < 70:
            if color_score < 50:
                recommendations.append("Consider using colors that more closely match the target environment")
            else:
                recommendations.append("Fine-tune color matching - consider adjusting hue and saturation")
        
        # Pattern-based recommendations  
        pattern_score = component_scores.get('pattern', 0)
        if pattern_score < 70:
            if pattern_score < 50:
                recommendations.append("Add more disruptive pattern elements to break up recognizable shapes")
            else:
                recommendations.append("Enhance pattern complexity to improve shape concealment")
        
        # Brightness-based recommendations
        brightness_score = component_scores.get('brightness', 0)
        if brightness_score < 70:
            if brightness_score < 50:
                recommendations.append("Adjust brightness levels to better match environmental lighting")
            else:
                recommendations.append("Consider local contrast adjustments for better brightness blending")
        
        # Distance-based recommendations
        distance_score = component_scores.get('distance', 0)
        if distance_score < 70:
            recommendations.append("Improve effectiveness at longer distances with enhanced pattern scaling")
        
        # Overall recommendations
        overall_score = aggregated_results.get('overall_score', 0)
        if overall_score > 85:
            recommendations.append("Excellent camouflage effectiveness - consider testing in varied conditions")
        elif overall_score < 60:
            recommendations.append("Consider redesigning with focus on the lowest-scoring components")
        
        return recommendations
    
    def _create_analysis_summary(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics of the analysis"""
        component_scores = aggregated_results.get('component_scores', {})
        
        return {
            'best_component': max(component_scores.items(), key=lambda x: x[1]) if component_scores else ('none', 0),
            'worst_component': min(component_scores.items(), key=lambda x: x[1]) if component_scores else ('none', 0),
            'score_range': max(component_scores.values()) - min(component_scores.values()) if component_scores else 0,
            'components_analyzed': len(component_scores),
            'pipelines_executed': len(aggregated_results.get('analysis_metadata', {}).get('pipelines_executed', [])),
            'errors_encountered': len(aggregated_results.get('pipeline_errors', []))
        }