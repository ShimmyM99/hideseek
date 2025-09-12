import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import json
import os

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('environment_analyzer')


class EnvironmentalContextAnalyzer:
    """
    Advanced environmental context analysis for camouflage effectiveness.
    Analyzes compatibility across different environments and seasonal variations.
    """
    
    def __init__(self):
        self.environment_configs = config.get('environments', {})
        self.analysis_params = config.get_analysis_params()
        
        # Load environment database
        self.environment_database = self._load_environment_database()
        
        # Seasonal variation parameters
        self.seasons = ['spring', 'summer', 'autumn', 'winter']
        self.seasonal_adjustments = {
            'spring': {'hue_shift': 10, 'saturation_mult': 1.2, 'brightness_add': 5},
            'summer': {'hue_shift': -5, 'saturation_mult': 1.3, 'brightness_add': 10},
            'autumn': {'hue_shift': -15, 'saturation_mult': 0.9, 'brightness_add': -5},
            'winter': {'hue_shift': 5, 'saturation_mult': 0.7, 'brightness_add': -10}
        }
        
        logger.info(f"EnvironmentAnalyzer initialized with {len(self.environment_configs)} environments")
    
    def analyze_environmental_context(self, camo_img: np.ndarray,
                                     options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive environmental context analysis.
        
        Args:
            camo_img: Camouflage image in BGR format
            options: Analysis options
            
        Returns:
            Environmental context analysis results
        """
        start_time = cv2.getTickCount()
        logger.info("Starting environmental context analysis")
        
        if options is None:
            options = {}
        
        try:
            # Step 1: Primary environment classification
            logger.debug("Step 1: Environment classification")
            primary_environment = self._classify_primary_environment(camo_img)
            
            # Step 2: Multi-environment compatibility testing
            logger.debug("Step 2: Multi-environment compatibility")
            environment_compatibility = self._test_environment_compatibility(camo_img, options)
            
            # Step 3: Seasonal variation analysis
            logger.debug("Step 3: Seasonal variation analysis")
            seasonal_analysis = self._analyze_seasonal_variations(camo_img, primary_environment, options)
            
            # Step 4: Versatility matrix generation
            logger.debug("Step 4: Versatility matrix")
            versatility_matrix = self._generate_versatility_matrix(camo_img, options)
            
            # Step 5: Background complexity analysis
            logger.debug("Step 5: Background complexity")
            complexity_analysis = self._analyze_background_complexity(camo_img, primary_environment)
            
            # Step 6: Environmental match scoring
            logger.debug("Step 6: Environmental match scoring")
            match_scores = self._compute_environmental_match_scores(
                camo_img, environment_compatibility, seasonal_analysis
            )
            
            # Step 7: Calculate overall environmental adaptability score
            environmental_score = self._calculate_environmental_adaptability_score({
                'environment_compatibility': environment_compatibility,
                'seasonal_analysis': seasonal_analysis,
                'versatility_matrix': versatility_matrix,
                'complexity_analysis': complexity_analysis,
                'match_scores': match_scores
            })
            
            execution_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            results = {
                'pipeline_name': 'environmental_context',
                'score': environmental_score,
                'primary_environment': primary_environment,
                'environment_compatibility': environment_compatibility,
                'seasonal_analysis': seasonal_analysis,
                'versatility_matrix': versatility_matrix,
                'complexity_analysis': complexity_analysis,
                'match_scores': match_scores,
                'execution_time': execution_time,
                'metadata': {
                    'environments_tested': list(self.environment_configs.keys()),
                    'seasons_analyzed': self.seasons if options.get('seasonal_analysis', False) else [],
                    'primary_environment_confidence': primary_environment.get('confidence', 0)
                }
            }
            
            logger.info(f"Environmental analysis completed: score={environmental_score:.1f}/100 in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Environmental analysis failed: {str(e)}")
            return self._create_error_result(str(e))
    
    def _load_environment_database(self) -> Dict[str, Any]:
        """Load or create environment reference database"""
        
        # Try to load existing database
        try:
            from ..core.data_manager import TestDataManager
            data_manager = TestDataManager()
            return data_manager.load_environment_database()
        except:
            # Create basic database from config
            database = {}
            for env_type, config_data in self.environment_configs.items():
                database[env_type] = {
                    'color_profiles': self._create_color_profile_from_config(config_data),
                    'texture_characteristics': config_data.get('texture_complexity', 0.5),
                    'lighting_patterns': config_data.get('lighting_characteristics', []),
                    'reference_images': []  # Would be populated with actual reference images
                }
            return database
    
    def _create_color_profile_from_config(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create color profile from environment configuration"""
        
        primary_colors = env_config.get('primary_colors', ['#808080'])
        
        # Convert hex colors to BGR
        bgr_colors = []
        for hex_color in primary_colors:
            # Remove '#' if present
            hex_color = hex_color.lstrip('#')
            # Convert hex to RGB, then to BGR
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])  # RGB to BGR
            bgr_colors.append(bgr)
        
        return {
            'dominant_colors_bgr': bgr_colors,
            'color_distribution': self._estimate_color_distribution(bgr_colors),
            'color_variance': self._calculate_color_variance(bgr_colors)
        }
    
    def _estimate_color_distribution(self, colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """Estimate statistical distribution of colors"""
        
        if not colors:
            return {'mean_bgr': [128, 128, 128], 'std_bgr': [0, 0, 0]}
        
        colors_array = np.array(colors)
        
        return {
            'mean_bgr': colors_array.mean(axis=0).tolist(),
            'std_bgr': colors_array.std(axis=0).tolist(),
            'color_range': (colors_array.max(axis=0) - colors_array.min(axis=0)).tolist()
        }
    
    def _calculate_color_variance(self, colors: List[Tuple[int, int, int]]) -> float:
        """Calculate overall color variance"""
        
        if len(colors) < 2:
            return 0.0
        
        colors_array = np.array(colors)
        
        # Calculate pairwise distances
        distances = pdist(colors_array)
        return float(np.mean(distances))
    
    def _classify_primary_environment(self, camo_img: np.ndarray) -> Dict[str, Any]:
        """Classify the primary environment type for the camouflage"""
        
        # Extract image features for classification
        features = self._extract_environment_features(camo_img)
        
        # Score against each environment type
        environment_scores = {}
        
        for env_type, env_database in self.environment_database.items():
            score = self._calculate_environment_similarity_score(features, env_database)
            environment_scores[env_type] = score
        
        # Find best match
        best_environment = max(environment_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence (ratio of best to second-best)
        sorted_scores = sorted(environment_scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1] > 0:
            confidence = sorted_scores[0] / sorted_scores[1]
        else:
            confidence = 1.0
        
        return {
            'environment': best_environment[0],
            'score': float(best_environment[1]),
            'confidence': float(min(confidence, 2.0)),  # Cap confidence
            'all_scores': {k: float(v) for k, v in environment_scores.items()}
        }
    
    def _extract_environment_features(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract features for environment classification"""
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Color features
        color_features = self._extract_color_features(img, hsv, lab)
        
        # Texture features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texture_features = self._extract_texture_features(gray)
        
        # Brightness features
        brightness_features = self._extract_brightness_features(lab[:, :, 0])
        
        return {
            'color_features': color_features,
            'texture_features': texture_features,
            'brightness_features': brightness_features
        }
    
    def _extract_color_features(self, bgr: np.ndarray, hsv: np.ndarray, 
                               lab: np.ndarray) -> Dict[str, Any]:
        """Extract color-based features"""
        
        # Dominant colors using K-means
        pixels = bgr.reshape(-1, 3)
        k = min(8, len(np.unique(pixels.reshape(-1))))  # Adaptive number of clusters
        
        if k > 1:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)
        else:
            dominant_colors = [np.mean(pixels, axis=0).astype(int)]
        
        # Color distribution statistics
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1] 
        v_channel = hsv[:, :, 2]
        
        # Hue distribution (circular statistics)
        hue_hist, _ = np.histogram(h_channel, bins=36, range=(0, 180))
        hue_hist = hue_hist / np.sum(hue_hist)
        dominant_hue_ranges = self._find_dominant_hue_ranges(hue_hist)
        
        return {
            'dominant_colors_bgr': dominant_colors.tolist(),
            'dominant_hue_ranges': dominant_hue_ranges,
            'saturation_stats': {
                'mean': float(np.mean(s_channel)),
                'std': float(np.std(s_channel)),
                'distribution': np.histogram(s_channel, bins=10, density=True)[0].tolist()
            },
            'value_stats': {
                'mean': float(np.mean(v_channel)),
                'std': float(np.std(v_channel)),
                'distribution': np.histogram(v_channel, bins=10, density=True)[0].tolist()
            },
            'color_diversity': self._calculate_color_diversity_index(bgr)
        }
    
    def _find_dominant_hue_ranges(self, hue_hist: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int]]:
        """Find dominant hue ranges from histogram"""
        
        # Smooth the histogram to find peaks
        smoothed = np.convolve(hue_hist, np.ones(3)/3, mode='same')
        
        # Find peaks above threshold
        peaks = []
        for i in range(len(smoothed)):
            if smoothed[i] > threshold:
                peaks.append(i)
        
        if not peaks:
            return [(0, 180)]  # Full range if no clear peaks
        
        # Group consecutive peaks into ranges
        ranges = []
        start = peaks[0]
        
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] > 1:  # Gap found
                end = peaks[i-1]
                ranges.append((start * 5, end * 5))  # Convert bin to hue value
                start = peaks[i]
        
        # Add final range
        ranges.append((start * 5, peaks[-1] * 5))
        
        return ranges
    
    def _calculate_color_diversity_index(self, img: np.ndarray) -> float:
        """Calculate color diversity index (Shannon entropy of color histogram)"""
        
        # Convert to simpler color space for histogram
        img_simplified = (img // 32) * 32  # Reduce color levels
        
        # Calculate 3D histogram
        hist = cv2.calcHist([img_simplified], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy_value = entropy(hist + 1e-10)  # Add small value to avoid log(0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(8**3)
        normalized_entropy = entropy_value / max_entropy
        
        return float(normalized_entropy)
    
    def _extract_texture_features(self, gray: np.ndarray) -> Dict[str, Any]:
        """Extract texture-based features"""
        
        # Gradient-based texture energy
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_energy = np.mean(gradient_magnitude) / 255.0
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Local binary pattern approximation
        texture_variance = self._calculate_texture_variance(gray)
        
        return {
            'texture_energy': float(texture_energy),
            'edge_density': float(edge_density),
            'texture_variance': float(texture_variance),
            'complexity_score': float((texture_energy + edge_density + texture_variance) / 3)
        }
    
    def _calculate_texture_variance(self, gray: np.ndarray) -> float:
        """Calculate local texture variance"""
        
        # Use local standard deviation as texture measure
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        # Calculate local mean
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Calculate local variance
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        return np.mean(local_variance) / (255.0**2)
    
    def _extract_brightness_features(self, luminance: np.ndarray) -> Dict[str, Any]:
        """Extract brightness-based features"""
        
        return {
            'mean_brightness': float(np.mean(luminance)),
            'brightness_std': float(np.std(luminance)),
            'brightness_range': float(np.max(luminance) - np.min(luminance)),
            'brightness_distribution': np.histogram(luminance, bins=10, density=True)[0].tolist(),
            'brightness_entropy': float(entropy(np.histogram(luminance, bins=256, density=True)[0] + 1e-10))
        }
    
    def _calculate_environment_similarity_score(self, features: Dict[str, Any], 
                                               env_database: Dict[str, Any]) -> float:
        """Calculate similarity score to specific environment"""
        
        scores = []
        
        # Color similarity
        color_score = self._calculate_color_similarity(
            features['color_features'], env_database.get('color_profiles', {})
        )
        scores.append(color_score)
        
        # Texture similarity
        texture_score = self._calculate_texture_similarity_to_env(
            features['texture_features'], env_database.get('texture_characteristics', 0.5)
        )
        scores.append(texture_score)
        
        # Brightness similarity
        brightness_score = self._calculate_brightness_similarity_to_env(
            features['brightness_features'], env_database
        )
        scores.append(brightness_score)
        
        # Weighted average
        weights = [0.5, 0.3, 0.2]  # Color most important, then texture, then brightness
        return sum(s * w for s, w in zip(scores, weights))
    
    def _calculate_color_similarity(self, color_features: Dict[str, Any], 
                                   color_profile: Dict[str, Any]) -> float:
        """Calculate color similarity to environment color profile"""
        
        if not color_profile or 'dominant_colors_bgr' not in color_profile:
            return 0.5  # Neutral score if no profile
        
        camo_colors = np.array(color_features['dominant_colors_bgr'])
        env_colors = np.array(color_profile['dominant_colors_bgr'])
        
        if len(camo_colors) == 0 or len(env_colors) == 0:
            return 0.5
        
        # Calculate minimum distance from each camo color to environment colors
        distances = cdist(camo_colors, env_colors)
        min_distances = np.min(distances, axis=1)
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        max_possible_distance = np.sqrt(3 * 255**2)  # Maximum distance in RGB space
        similarities = 1.0 - (min_distances / max_possible_distance)
        
        return float(np.mean(similarities))
    
    def _calculate_texture_similarity_to_env(self, texture_features: Dict[str, Any], 
                                            env_texture_complexity: float) -> float:
        """Calculate texture similarity to environment"""
        
        camo_complexity = texture_features.get('complexity_score', 0.5)
        
        # Similarity based on how close the complexity scores are
        complexity_diff = abs(camo_complexity - env_texture_complexity)
        similarity = max(0, 1.0 - complexity_diff)
        
        return float(similarity)
    
    def _calculate_brightness_similarity_to_env(self, brightness_features: Dict[str, Any],
                                               env_database: Dict[str, Any]) -> float:
        """Calculate brightness similarity to environment"""
        
        # Use general brightness characteristics
        # Most environments have similar brightness ranges, so this is less discriminative
        camo_brightness = brightness_features.get('mean_brightness', 128)
        
        # Normalize brightness (0-255 range)
        normalized_brightness = camo_brightness / 255.0
        
        # Most environments are compatible with mid-range brightness
        # Penalize extreme brightness (too dark or too bright)
        if 0.2 <= normalized_brightness <= 0.8:
            return 1.0
        elif 0.1 <= normalized_brightness <= 0.9:
            return 0.7
        else:
            return 0.3
    
    def _test_environment_compatibility(self, camo_img: np.ndarray, 
                                      options: Dict[str, Any]) -> Dict[str, Any]:
        """Test compatibility across multiple environments"""
        
        compatibility_scores = {}
        detailed_analysis = {}
        
        for env_type, env_config in self.environment_configs.items():
            logger.debug(f"Testing compatibility with {env_type}")
            
            try:
                # Extract features for this environment test
                features = self._extract_environment_features(camo_img)
                
                # Calculate environment-specific compatibility
                env_database = self.environment_database.get(env_type, {})
                compatibility = self._calculate_environment_similarity_score(features, env_database)
                
                compatibility_scores[env_type] = float(compatibility * 100)
                
                # Detailed analysis for this environment
                detailed_analysis[env_type] = self._detailed_environment_analysis(
                    camo_img, env_type, env_config
                )
                
            except Exception as e:
                logger.warning(f"Environment compatibility test failed for {env_type}: {e}")
                compatibility_scores[env_type] = 0.0
                detailed_analysis[env_type] = {'error': str(e)}
        
        return {
            'compatibility_scores': compatibility_scores,
            'detailed_analysis': detailed_analysis,
            'best_environment': max(compatibility_scores.items(), key=lambda x: x[1])[0] if compatibility_scores else 'unknown',
            'average_compatibility': float(np.mean(list(compatibility_scores.values()))) if compatibility_scores else 0.0
        }
    
    def _detailed_environment_analysis(self, camo_img: np.ndarray, env_type: str, 
                                      env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis for specific environment"""
        
        # Color analysis
        target_colors = env_config.get('primary_colors', [])
        color_match = self._analyze_color_match_to_environment(camo_img, target_colors)
        
        # Lighting analysis
        lighting_chars = env_config.get('lighting_characteristics', [])
        lighting_match = self._analyze_lighting_compatibility(camo_img, lighting_chars)
        
        # Texture complexity analysis
        target_complexity = env_config.get('texture_complexity', 0.5)
        texture_match = self._analyze_texture_match(camo_img, target_complexity)
        
        return {
            'color_match_score': color_match,
            'lighting_compatibility': lighting_match,
            'texture_match_score': texture_match,
            'overall_match': float((color_match + lighting_match + texture_match) / 3 * 100)
        }
    
    def _analyze_color_match_to_environment(self, img: np.ndarray, 
                                          target_colors: List[str]) -> float:
        """Analyze how well image colors match environment target colors"""
        
        if not target_colors:
            return 0.5
        
        # Convert target colors to BGR
        target_bgr = []
        for hex_color in target_colors:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            bgr = np.array([rgb[2], rgb[1], rgb[0]])  # RGB to BGR
            target_bgr.append(bgr)
        
        target_bgr = np.array(target_bgr)
        
        # Sample image colors
        img_colors = img.reshape(-1, 3)
        
        # Find closest matches
        if len(target_bgr) > 0:
            distances = cdist(img_colors, target_bgr)
            min_distances = np.min(distances, axis=1)
            
            # Convert to similarity scores
            max_distance = np.sqrt(3 * 255**2)
            similarities = 1.0 - (min_distances / max_distance)
            
            return float(np.mean(similarities))
        
        return 0.5
    
    def _analyze_lighting_compatibility(self, img: np.ndarray, 
                                       lighting_chars: List[str]) -> float:
        """Analyze lighting compatibility"""
        
        if not lighting_chars:
            return 0.5
        
        # Extract lighting characteristics from image
        luminance = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
        
        # Analyze brightness distribution
        brightness_mean = np.mean(luminance)
        brightness_std = np.std(luminance)
        
        compatibility_score = 0.5  # Default neutral score
        
        for char in lighting_chars:
            if char == 'bright':
                # Prefer higher brightness
                if brightness_mean > 150:
                    compatibility_score = max(compatibility_score, 0.8)
            elif char == 'low' or char == 'filtered':
                # Prefer lower brightness
                if brightness_mean < 120:
                    compatibility_score = max(compatibility_score, 0.8)
            elif char == 'dappled' or char == 'mixed':
                # Prefer high variance in brightness
                if brightness_std > 40:
                    compatibility_score = max(compatibility_score, 0.8)
            elif char == 'uniform':
                # Prefer low variance
                if brightness_std < 20:
                    compatibility_score = max(compatibility_score, 0.8)
        
        return compatibility_score
    
    def _analyze_texture_match(self, img: np.ndarray, target_complexity: float) -> float:
        """Analyze texture complexity match"""
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texture_features = self._extract_texture_features(gray)
        actual_complexity = texture_features['complexity_score']
        
        # Calculate similarity to target complexity
        complexity_diff = abs(actual_complexity - target_complexity)
        similarity = max(0, 1.0 - complexity_diff)
        
        return float(similarity)
    
    def _analyze_seasonal_variations(self, camo_img: np.ndarray, primary_environment: Dict[str, Any],
                                    options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze effectiveness across seasonal variations"""
        
        if not options.get('seasonal_analysis', False):
            return {'seasonal_analysis_skipped': True}
        
        seasonal_results = {}
        
        for season in self.seasons:
            logger.debug(f"Analyzing {season} variation")
            
            try:
                # Apply seasonal color adjustments
                season_adjusted_img = self._apply_seasonal_adjustment(camo_img, season)
                
                # Test compatibility with primary environment using adjusted image
                env_type = primary_environment.get('environment', 'woodland')
                env_database = self.environment_database.get(env_type, {})
                
                features = self._extract_environment_features(season_adjusted_img)
                seasonal_compatibility = self._calculate_environment_similarity_score(features, env_database)
                
                seasonal_results[season] = {
                    'compatibility_score': float(seasonal_compatibility * 100),
                    'color_shift_applied': self.seasonal_adjustments[season],
                    'effectiveness': self._categorize_seasonal_effectiveness(seasonal_compatibility)
                }
                
            except Exception as e:
                logger.warning(f"Seasonal analysis failed for {season}: {e}")
                seasonal_results[season] = {'error': str(e)}
        
        # Calculate seasonal consistency
        valid_scores = [r['compatibility_score'] for r in seasonal_results.values() 
                       if 'compatibility_score' in r]
        
        if valid_scores:
            seasonal_consistency = 1.0 - (np.std(valid_scores) / 100.0)  # Normalize by score range
            best_season = max([(k, v['compatibility_score']) for k, v in seasonal_results.items() 
                             if 'compatibility_score' in v], key=lambda x: x[1])[0]
            worst_season = min([(k, v['compatibility_score']) for k, v in seasonal_results.items() 
                              if 'compatibility_score' in v], key=lambda x: x[1])[0]
        else:
            seasonal_consistency = 0.0
            best_season = worst_season = 'unknown'
        
        return {
            'seasonal_results': seasonal_results,
            'seasonal_consistency': float(max(0, seasonal_consistency)),
            'best_season': best_season,
            'worst_season': worst_season,
            'average_seasonal_score': float(np.mean(valid_scores)) if valid_scores else 0.0
        }
    
    def _apply_seasonal_adjustment(self, img: np.ndarray, season: str) -> np.ndarray:
        """Apply seasonal color adjustments to simulate environmental changes"""
        
        adjustments = self.seasonal_adjustments.get(season, {})
        
        # Convert to HSV for easier manipulation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Apply hue shift
        hue_shift = adjustments.get('hue_shift', 0)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Apply saturation multiplier
        sat_mult = adjustments.get('saturation_mult', 1.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
        
        # Apply brightness adjustment
        brightness_add = adjustments.get('brightness_add', 0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness_add, 0, 255)
        
        # Convert back to BGR
        adjusted_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return adjusted_img
    
    def _categorize_seasonal_effectiveness(self, score: float) -> str:
        """Categorize seasonal effectiveness"""
        
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.65:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_versatility_matrix(self, camo_img: np.ndarray, 
                                    options: Dict[str, Any]) -> Dict[str, Any]:
        """Generate versatility matrix across multiple environments"""
        
        if not options.get('test_multiple_environments', True):
            return {'versatility_analysis_skipped': True}
        
        # Test against all environment types
        versatility_scores = {}
        
        for env_type in self.environment_configs.keys():
            env_database = self.environment_database.get(env_type, {})
            features = self._extract_environment_features(camo_img)
            score = self._calculate_environment_similarity_score(features, env_database)
            versatility_scores[env_type] = float(score * 100)
        
        # Calculate versatility metrics
        scores = list(versatility_scores.values())
        versatility_metrics = {
            'average_score': float(np.mean(scores)),
            'score_consistency': float(1.0 - np.std(scores) / 100.0),  # Lower std = higher consistency
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'score_range': float(np.max(scores) - np.min(scores))
        }
        
        return {
            'environment_scores': versatility_scores,
            'versatility_metrics': versatility_metrics,
            'versatility_rating': self._rate_versatility(versatility_metrics)
        }
    
    def _rate_versatility(self, metrics: Dict[str, float]) -> str:
        """Rate overall versatility"""
        
        avg_score = metrics['average_score']
        consistency = metrics['score_consistency']
        
        # Combined score emphasizing both performance and consistency
        versatility_score = (avg_score * 0.7) + (consistency * 100 * 0.3)
        
        if versatility_score >= 75:
            return 'highly_versatile'
        elif versatility_score >= 60:
            return 'moderately_versatile'
        elif versatility_score >= 45:
            return 'limited_versatility'
        else:
            return 'environment_specific'
    
    def _analyze_background_complexity(self, camo_img: np.ndarray, 
                                      primary_environment: str) -> Dict[str, Any]:
        """Analyze background complexity requirements"""
        
        # Calculate image complexity
        complexity_metrics = self._calculate_image_complexity(camo_img)
        
        # Compare to environment requirements
        env_type = primary_environment.get('environment', 'woodland')
        target_complexity = self.environment_configs.get(env_type, {}).get('texture_complexity', 0.5)
        
        complexity_match = 1.0 - abs(complexity_metrics['overall_complexity'] - target_complexity)
        
        return {
            'image_complexity': complexity_metrics,
            'target_complexity': float(target_complexity),
            'complexity_match_score': float(max(0, complexity_match) * 100),
            'complexity_assessment': self._assess_complexity_level(complexity_metrics['overall_complexity'])
        }
    
    def _calculate_image_complexity(self, img: np.ndarray) -> Dict[str, float]:
        """Calculate various complexity measures"""
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge-based complexity
        edges = cv2.Canny(gray, 50, 150)
        edge_complexity = np.sum(edges > 0) / edges.size
        
        # Gradient-based complexity
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_complexity = np.mean(gradient_magnitude) / 255.0
        
        # Entropy-based complexity
        hist = np.histogram(gray, bins=256, density=True)[0]
        entropy_complexity = entropy(hist + 1e-10) / np.log(256)
        
        # Overall complexity (weighted average)
        overall_complexity = (edge_complexity * 0.4 + 
                            gradient_complexity * 0.4 + 
                            entropy_complexity * 0.2)
        
        return {
            'edge_complexity': float(edge_complexity),
            'gradient_complexity': float(gradient_complexity),
            'entropy_complexity': float(entropy_complexity),
            'overall_complexity': float(overall_complexity)
        }
    
    def _assess_complexity_level(self, complexity: float) -> str:
        """Assess complexity level"""
        
        if complexity >= 0.7:
            return 'high_complexity'
        elif complexity >= 0.4:
            return 'medium_complexity'
        else:
            return 'low_complexity'
    
    def _compute_environmental_match_scores(self, camo_img: np.ndarray, 
                                           environment_compatibility: Dict[str, Any],
                                           seasonal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive environmental match scores"""
        
        # Base compatibility scores
        base_scores = environment_compatibility.get('compatibility_scores', {})
        
        # Seasonal adjustment factors
        seasonal_factors = {}
        if 'seasonal_results' in seasonal_analysis:
            for season, results in seasonal_analysis['seasonal_results'].items():
                if 'compatibility_score' in results:
                    seasonal_factors[season] = results['compatibility_score'] / 100.0
        
        # Calculate adjusted scores
        adjusted_scores = {}
        for env, base_score in base_scores.items():
            if seasonal_factors:
                # Apply seasonal adjustment
                seasonal_avg = np.mean(list(seasonal_factors.values()))
                adjusted_score = base_score * seasonal_avg
            else:
                adjusted_score = base_score
            
            adjusted_scores[env] = float(adjusted_score)
        
        return {
            'base_scores': base_scores,
            'seasonal_factors': seasonal_factors,
            'adjusted_scores': adjusted_scores,
            'best_environment_match': max(adjusted_scores.items(), key=lambda x: x[1])[0] if adjusted_scores else 'unknown'
        }
    
    def _calculate_environmental_adaptability_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall environmental adaptability score"""
        
        scores = []
        weights = []
        
        # Environment compatibility score
        env_compatibility = analysis_results.get('environment_compatibility', {})
        avg_compatibility = env_compatibility.get('average_compatibility', 0)
        scores.append(avg_compatibility)
        weights.append(0.35)
        
        # Versatility matrix score
        versatility_matrix = analysis_results.get('versatility_matrix', {})
        if 'versatility_metrics' in versatility_matrix:
            versatility_score = versatility_matrix['versatility_metrics'].get('average_score', 0)
            scores.append(versatility_score)
            weights.append(0.25)
        
        # Seasonal consistency score
        seasonal_analysis = analysis_results.get('seasonal_analysis', {})
        if 'seasonal_consistency' in seasonal_analysis:
            seasonal_score = seasonal_analysis['seasonal_consistency'] * 100
            scores.append(seasonal_score)
            weights.append(0.20)
        
        # Background complexity match score
        complexity_analysis = analysis_results.get('complexity_analysis', {})
        complexity_score = complexity_analysis.get('complexity_match_score', 50)
        scores.append(complexity_score)
        weights.append(0.20)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                final_score = sum(s * w for s, w in zip(scores, normalized_weights))
            else:
                final_score = np.mean(scores)
        else:
            final_score = 0.0
        
        return max(0.0, min(100.0, final_score))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            'pipeline_name': 'environmental_context',
            'error': error_message,
            'score': 0.0,
            'execution_time': 0.0
        }