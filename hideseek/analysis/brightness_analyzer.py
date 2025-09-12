import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy import ndimage
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('brightness_analyzer')


class BrightnessContrastAnalyzer:
    """
    Advanced brightness and contrast matching analysis for camouflage effectiveness.
    Analyzes luminance matching, local contrast, and illumination adaptation.
    """
    
    def __init__(self):
        self.lighting_configs = config.get_lighting_config()
        self.analysis_params = config.get_analysis_params()
        
        # Standard illumination conditions
        self.illumination_conditions = ['daylight', 'twilight', 'night', 'infrared']
        
        # Local contrast window sizes
        self.contrast_windows = [8, 16, 32, 64]
        
        logger.info("BrightnessAnalyzer initialized with multi-illumination testing")
    
    def analyze_brightness_contrast(self, camo_img: np.ndarray, bg_img: np.ndarray = None,
                                   options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive brightness and contrast analysis.
        
        Args:
            camo_img: Camouflage image in BGR format
            bg_img: Optional background reference image
            options: Analysis options
            
        Returns:
            Brightness and contrast analysis results
        """
        start_time = cv2.getTickCount()
        logger.info("Starting brightness and contrast analysis")
        
        if options is None:
            options = {}
        
        try:
            # Step 1: Extract luminance channels
            logger.debug("Step 1: Extracting luminance channels")
            camo_luminance = self.extract_luminance_channel(camo_img)
            bg_luminance = self.extract_luminance_channel(bg_img) if bg_img is not None else None
            
            # Step 2: Basic brightness analysis
            logger.debug("Step 2: Basic brightness analysis")
            brightness_analysis = self._analyze_brightness_distribution(camo_luminance, bg_luminance)
            
            # Step 3: Local contrast analysis
            logger.debug("Step 3: Local contrast analysis")
            contrast_analysis = self._analyze_local_contrast(camo_luminance, bg_luminance, options)
            
            # Step 4: Shadow pattern analysis
            logger.debug("Step 4: Shadow pattern analysis")
            shadow_analysis = self._analyze_shadow_patterns(camo_img, camo_luminance, options)
            
            # Step 5: Multi-illumination testing
            logger.debug("Step 5: Multi-illumination testing")
            illumination_analysis = self._test_under_multiple_illuminations(camo_img, options)
            
            # Step 6: Histogram analysis
            logger.debug("Step 6: Histogram analysis")
            histogram_analysis = self._analyze_brightness_histograms(camo_luminance, bg_luminance)
            
            # Step 7: Adaptive brightness analysis
            logger.debug("Step 7: Adaptive brightness analysis")
            adaptive_analysis = self._analyze_adaptive_brightness(camo_luminance, bg_luminance, options)
            
            # Step 8: Calculate final brightness matching score
            brightness_score = self._calculate_brightness_blend_score({
                'brightness_analysis': brightness_analysis,
                'contrast_analysis': contrast_analysis,
                'shadow_analysis': shadow_analysis,
                'illumination_analysis': illumination_analysis,
                'histogram_analysis': histogram_analysis,
                'adaptive_analysis': adaptive_analysis
            })
            
            execution_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            results = {
                'pipeline_name': 'brightness_matching',
                'score': brightness_score,
                'brightness_analysis': brightness_analysis,
                'contrast_analysis': contrast_analysis,
                'shadow_analysis': shadow_analysis,
                'illumination_analysis': illumination_analysis,
                'histogram_analysis': histogram_analysis,
                'adaptive_analysis': adaptive_analysis,
                'execution_time': execution_time,
                'metadata': {
                    'illumination_conditions_tested': len(self.illumination_conditions),
                    'contrast_window_sizes': self.contrast_windows,
                    'analysis_method': 'multi_scale_luminance'
                }
            }
            
            logger.info(f"Brightness analysis completed: score={brightness_score:.1f}/100 in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Brightness analysis failed: {str(e)}")
            return self._create_error_result(str(e))
    
    def extract_luminance_channel(self, img: np.ndarray) -> np.ndarray:
        """
        Extract perceptual luminance channel from BGR image.
        
        Args:
            img: BGR image
            
        Returns:
            Luminance channel (Y in YUV or L in LAB)
        """
        if img is None:
            return None
        
        # Convert to LAB color space for perceptual luminance
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0]  # L channel (0-255)
        
        return luminance
    
    def _analyze_brightness_distribution(self, camo_luminance: np.ndarray, 
                                        bg_luminance: np.ndarray = None) -> Dict[str, Any]:
        """Analyze brightness distribution statistics"""
        
        # Calculate basic statistics for camouflage image
        camo_stats = {
            'mean': float(np.mean(camo_luminance)),
            'std': float(np.std(camo_luminance)),
            'min': float(np.min(camo_luminance)),
            'max': float(np.max(camo_luminance)),
            'median': float(np.median(camo_luminance)),
            'range': float(np.max(camo_luminance) - np.min(camo_luminance)),
            'percentile_25': float(np.percentile(camo_luminance, 25)),
            'percentile_75': float(np.percentile(camo_luminance, 75))
        }
        
        result = {'camo_stats': camo_stats}
        
        # If background provided, analyze matching
        if bg_luminance is not None:
            # Ensure same size
            if bg_luminance.shape != camo_luminance.shape:
                bg_luminance = cv2.resize(bg_luminance, 
                                        (camo_luminance.shape[1], camo_luminance.shape[0]))
            
            bg_stats = {
                'mean': float(np.mean(bg_luminance)),
                'std': float(np.std(bg_luminance)),
                'min': float(np.min(bg_luminance)),
                'max': float(np.max(bg_luminance)),
                'median': float(np.median(bg_luminance)),
                'range': float(np.max(bg_luminance) - np.min(bg_luminance)),
                'percentile_25': float(np.percentile(bg_luminance, 25)),
                'percentile_75': float(np.percentile(bg_luminance, 75))
            }
            
            # Calculate brightness matching metrics
            mean_diff = abs(camo_stats['mean'] - bg_stats['mean'])
            std_ratio = min(camo_stats['std'], bg_stats['std']) / max(camo_stats['std'], bg_stats['std'])
            range_ratio = min(camo_stats['range'], bg_stats['range']) / max(camo_stats['range'], bg_stats['range'])
            
            # Brightness matching score (0-100)
            brightness_matching = max(0, 100 - (mean_diff / 255 * 100))
            distribution_matching = (std_ratio + range_ratio) / 2 * 100
            
            result.update({
                'bg_stats': bg_stats,
                'brightness_matching_score': float(brightness_matching),
                'distribution_matching_score': float(distribution_matching),
                'mean_difference': float(mean_diff),
                'std_ratio': float(std_ratio),
                'range_ratio': float(range_ratio)
            })
        
        return result
    
    def _analyze_local_contrast(self, camo_luminance: np.ndarray, 
                               bg_luminance: np.ndarray = None,
                               options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze local contrast at multiple scales"""
        
        result = {'camo_contrast': {}, 'multi_scale_analysis': {}}
        
        # Calculate local contrast for each window size
        for window_size in self.contrast_windows:
            contrast_map = self._calculate_local_contrast(camo_luminance, window_size)
            
            # Statistical analysis of contrast map
            contrast_stats = {
                'mean': float(np.mean(contrast_map)),
                'std': float(np.std(contrast_map)),
                'max': float(np.max(contrast_map)),
                'energy': float(np.sum(contrast_map**2))
            }
            
            result['camo_contrast'][f'window_{window_size}'] = contrast_stats
        
        # Multi-scale contrast energy
        all_contrast_means = [stats['mean'] for stats in result['camo_contrast'].values()]
        result['multi_scale_analysis']['contrast_consistency'] = float(1.0 - np.std(all_contrast_means) / (np.mean(all_contrast_means) + 1e-6))
        
        # If background provided, compare contrast patterns
        if bg_luminance is not None:
            result['bg_contrast'] = {}
            contrast_similarities = []
            
            for window_size in self.contrast_windows:
                bg_contrast_map = self._calculate_local_contrast(bg_luminance, window_size)
                camo_contrast_map = self._calculate_local_contrast(camo_luminance, window_size)
                
                bg_contrast_stats = {
                    'mean': float(np.mean(bg_contrast_map)),
                    'std': float(np.std(bg_contrast_map)),
                    'max': float(np.max(bg_contrast_map)),
                    'energy': float(np.sum(bg_contrast_map**2))
                }
                
                result['bg_contrast'][f'window_{window_size}'] = bg_contrast_stats
                
                # Calculate contrast pattern similarity
                try:
                    # Normalize contrast maps
                    camo_norm = camo_contrast_map / (np.max(camo_contrast_map) + 1e-6)
                    bg_norm = bg_contrast_map / (np.max(bg_contrast_map) + 1e-6)
                    
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(camo_norm.flatten(), bg_norm.flatten())[0, 1]
                    if not np.isnan(correlation):
                        contrast_similarities.append(abs(correlation))
                except:
                    pass
            
            # Overall contrast similarity
            if contrast_similarities:
                result['contrast_similarity_score'] = float(np.mean(contrast_similarities) * 100)
            else:
                result['contrast_similarity_score'] = 0.0
        
        return result
    
    def _calculate_local_contrast(self, luminance: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local contrast using sliding window"""
        
        # Use different contrast measures
        method = 'rms'  # Can be 'rms', 'michelson', 'weber'
        
        if method == 'rms':
            # RMS contrast (standard deviation in local window)
            kernel = np.ones((window_size, window_size)) / (window_size * window_size)
            local_mean = cv2.filter2D(luminance.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((luminance.astype(np.float32) - local_mean)**2, -1, kernel)
            contrast_map = np.sqrt(local_variance)
            
        elif method == 'michelson':
            # Michelson contrast: (max - min) / (max + min)
            kernel = np.ones((window_size, window_size))
            local_max = ndimage.maximum_filter(luminance.astype(np.float32), size=window_size)
            local_min = ndimage.minimum_filter(luminance.astype(np.float32), size=window_size)
            contrast_map = (local_max - local_min) / (local_max + local_min + 1e-6)
            
        else:  # weber
            # Weber contrast: (I - Ib) / Ib
            kernel = np.ones((window_size, window_size)) / (window_size * window_size)
            local_mean = cv2.filter2D(luminance.astype(np.float32), -1, kernel)
            contrast_map = np.abs(luminance.astype(np.float32) - local_mean) / (local_mean + 1e-6)
        
        return contrast_map
    
    def _analyze_shadow_patterns(self, camo_img: np.ndarray, camo_luminance: np.ndarray,
                                options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze shadow patterns and consistency"""
        
        # Detect potential shadow regions (darker areas with specific characteristics)
        shadow_mask = self._detect_shadow_regions(camo_luminance)
        
        # Analyze shadow properties
        if np.sum(shadow_mask) > 0:
            shadow_pixels = camo_luminance[shadow_mask > 0]
            non_shadow_pixels = camo_luminance[shadow_mask == 0]
            
            shadow_stats = {
                'shadow_coverage': float(np.sum(shadow_mask) / shadow_mask.size),
                'shadow_brightness_mean': float(np.mean(shadow_pixels)),
                'shadow_brightness_std': float(np.std(shadow_pixels)),
                'brightness_contrast_ratio': float(np.mean(non_shadow_pixels) / (np.mean(shadow_pixels) + 1e-6))
            }
            
            # Analyze shadow color temperature (using color image)
            shadow_color_analysis = self._analyze_shadow_color_temperature(camo_img, shadow_mask)
            shadow_stats.update(shadow_color_analysis)
            
        else:
            shadow_stats = {
                'shadow_coverage': 0.0,
                'shadow_brightness_mean': 0.0,
                'shadow_brightness_std': 0.0,
                'brightness_contrast_ratio': 1.0
            }
        
        # Shadow pattern regularity
        shadow_regularity = self._analyze_shadow_regularity(shadow_mask)
        shadow_stats['pattern_regularity'] = shadow_regularity
        
        # Shadow effectiveness score
        # Good camouflage should have realistic shadow patterns
        coverage_score = min(shadow_stats['shadow_coverage'] * 200, 100)  # Optimal around 50% coverage
        contrast_score = max(0, 100 - abs(shadow_stats['brightness_contrast_ratio'] - 2.0) * 25)  # Optimal ratio around 2:1
        regularity_score = (1.0 - shadow_stats['pattern_regularity']) * 100  # Less regular = more natural
        
        shadow_effectiveness = (coverage_score * 0.4 + contrast_score * 0.4 + regularity_score * 0.2)
        shadow_stats['shadow_effectiveness_score'] = float(max(0, min(100, shadow_effectiveness)))
        
        return shadow_stats
    
    def _detect_shadow_regions(self, luminance: np.ndarray) -> np.ndarray:
        """Detect potential shadow regions in luminance image"""
        
        # Use Otsu's thresholding to find darker regions
        _, shadow_mask = cv2.threshold(luminance, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Refine shadow mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        # Filter small regions
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(shadow_mask)
        
        min_area = luminance.size * 0.001  # Minimum 0.1% of image area
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.fillPoly(filtered_mask, [contour], 255)
        
        return filtered_mask
    
    def _analyze_shadow_color_temperature(self, color_img: np.ndarray, 
                                         shadow_mask: np.ndarray) -> Dict[str, Any]:
        """Analyze color temperature in shadow regions"""
        
        if np.sum(shadow_mask) == 0:
            return {'shadow_color_temperature': 'neutral', 'color_temperature_score': 50.0}
        
        # Extract shadow regions from color image
        shadow_pixels = color_img[shadow_mask > 0]
        non_shadow_pixels = color_img[shadow_mask == 0]
        
        if len(shadow_pixels) == 0 or len(non_shadow_pixels) == 0:
            return {'shadow_color_temperature': 'neutral', 'color_temperature_score': 50.0}
        
        # Calculate average BGR values
        shadow_mean = np.mean(shadow_pixels, axis=0)
        non_shadow_mean = np.mean(non_shadow_pixels, axis=0)
        
        # Analyze blue/red ratio (higher blue = cooler shadows, more realistic)
        shadow_blue_red_ratio = shadow_mean[0] / (shadow_mean[2] + 1e-6)  # B/R ratio
        non_shadow_blue_red_ratio = non_shadow_mean[0] / (non_shadow_mean[2] + 1e-6)
        
        # Realistic shadows should be cooler (more blue) than non-shadow areas
        color_temperature_difference = shadow_blue_red_ratio - non_shadow_blue_red_ratio
        
        if color_temperature_difference > 0.1:
            color_temperature = 'cool'
            temperature_score = min(100, color_temperature_difference * 200)
        elif color_temperature_difference < -0.1:
            color_temperature = 'warm' 
            temperature_score = max(0, 50 - abs(color_temperature_difference) * 200)
        else:
            color_temperature = 'neutral'
            temperature_score = 75.0
        
        return {
            'shadow_color_temperature': color_temperature,
            'color_temperature_score': float(temperature_score),
            'blue_red_ratio_difference': float(color_temperature_difference)
        }
    
    def _analyze_shadow_regularity(self, shadow_mask: np.ndarray) -> float:
        """Analyze regularity/randomness of shadow patterns"""
        
        if np.sum(shadow_mask) == 0:
            return 1.0  # Completely regular (no shadows)
        
        # Calculate spatial autocorrelation to measure pattern regularity
        # More random patterns have lower autocorrelation
        
        # Downsample for computational efficiency
        small_mask = cv2.resize(shadow_mask, (64, 64))
        
        # Calculate 2D autocorrelation using FFT
        fft_mask = np.fft.fft2(small_mask)
        autocorr = np.fft.ifft2(fft_mask * np.conj(fft_mask))
        autocorr = np.fft.fftshift(np.real(autocorr))
        
        # Normalize
        autocorr = autocorr / np.max(autocorr)
        
        # Calculate regularity as the sum of non-central autocorrelation values
        center = autocorr.shape[0] // 2
        autocorr[center, center] = 0  # Remove center peak
        
        regularity = np.sum(np.abs(autocorr)) / autocorr.size
        
        return min(1.0, regularity)
    
    def _test_under_multiple_illuminations(self, camo_img: np.ndarray, 
                                          options: Dict[str, Any]) -> Dict[str, Any]:
        """Test brightness adaptation under different illumination conditions"""
        
        results = {}
        
        for condition in self.illumination_conditions:
            if condition not in self.lighting_configs:
                continue
            
            try:
                # Simulate illumination condition
                adapted_img = self._simulate_illumination(camo_img, condition)
                
                # Extract luminance and analyze
                adapted_luminance = self.extract_luminance_channel(adapted_img)
                
                # Calculate adaptation metrics
                original_luminance = self.extract_luminance_channel(camo_img)
                adaptation_metrics = self._calculate_adaptation_metrics(
                    original_luminance, adapted_luminance
                )
                
                results[condition] = adaptation_metrics
                
            except Exception as e:
                logger.warning(f"Illumination test failed for {condition}: {e}")
                results[condition] = {'error': str(e)}
        
        # Calculate overall illumination adaptability score
        valid_scores = []
        for condition_results in results.values():
            if 'adaptation_score' in condition_results:
                valid_scores.append(condition_results['adaptation_score'])
        
        if valid_scores:
            overall_adaptability = np.mean(valid_scores)
        else:
            overall_adaptability = 0.0
        
        results['overall_adaptability_score'] = float(overall_adaptability)
        
        return results
    
    def _simulate_illumination(self, img: np.ndarray, condition: str) -> np.ndarray:
        """Simulate different illumination conditions"""
        
        config = self.lighting_configs.get(condition, {})
        
        if condition == 'daylight':
            # Neutral daylight (no change needed)
            return img.copy()
            
        elif condition == 'twilight':
            # Reduce overall brightness, add warm tint
            adapted = img.astype(np.float32)
            adapted *= 0.3  # Reduce brightness
            adapted[:, :, 0] *= 0.8  # Reduce blue
            adapted[:, :, 2] *= 1.2  # Increase red
            adapted = np.clip(adapted, 0, 255).astype(np.uint8)
            return adapted
            
        elif condition == 'night':
            # Very low brightness, cool tint
            adapted = img.astype(np.float32)
            adapted *= 0.05  # Very low brightness
            adapted[:, :, 0] *= 1.3  # Increase blue
            adapted[:, :, 2] *= 0.7  # Reduce red
            adapted = np.clip(adapted, 0, 255).astype(np.uint8)
            return adapted
            
        elif condition == 'infrared':
            # Convert to thermal-like representation
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Simple thermal simulation (warmer objects brighter)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            return thermal
            
        else:
            return img.copy()
    
    def _calculate_adaptation_metrics(self, original: np.ndarray, 
                                     adapted: np.ndarray) -> Dict[str, Any]:
        """Calculate adaptation metrics between original and adapted images"""
        
        # Brightness preservation
        original_mean = np.mean(original)
        adapted_mean = np.mean(adapted)
        brightness_change = abs(adapted_mean - original_mean) / 255.0
        brightness_preservation = max(0, 1.0 - brightness_change)
        
        # Contrast preservation
        original_contrast = np.std(original)
        adapted_contrast = np.std(adapted)
        contrast_change = abs(adapted_contrast - original_contrast) / 255.0
        contrast_preservation = max(0, 1.0 - contrast_change)
        
        # Detail preservation (using correlation)
        try:
            if original.shape == adapted.shape:
                correlation = np.corrcoef(original.flatten(), adapted.flatten())[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
        except:
            correlation = 0.0
        
        detail_preservation = max(0, correlation)
        
        # Overall adaptation score
        adaptation_score = (brightness_preservation * 0.3 + 
                          contrast_preservation * 0.3 + 
                          detail_preservation * 0.4) * 100
        
        return {
            'brightness_preservation': float(brightness_preservation * 100),
            'contrast_preservation': float(contrast_preservation * 100),
            'detail_preservation': float(detail_preservation * 100),
            'adaptation_score': float(max(0, min(100, adaptation_score)))
        }
    
    def _analyze_brightness_histograms(self, camo_luminance: np.ndarray,
                                      bg_luminance: np.ndarray = None) -> Dict[str, Any]:
        """Analyze brightness histograms and their properties"""
        
        # Calculate histogram for camouflage image
        camo_hist, _ = np.histogram(camo_luminance.ravel(), bins=256, range=(0, 255), density=True)
        
        # Calculate histogram properties
        camo_entropy = entropy(camo_hist + 1e-10)  # Add small value to avoid log(0)
        camo_uniformity = 1.0 - np.sum(camo_hist**2)
        camo_mean_brightness = np.sum(np.arange(256) * camo_hist)
        
        result = {
            'camo_histogram_properties': {
                'entropy': float(camo_entropy),
                'uniformity': float(camo_uniformity),
                'mean_brightness': float(camo_mean_brightness)
            }
        }
        
        # If background provided, compare histograms
        if bg_luminance is not None:
            bg_hist, _ = np.histogram(bg_luminance.ravel(), bins=256, range=(0, 255), density=True)
            
            # Calculate histogram similarity
            # Using multiple similarity measures
            similarities = self._calculate_histogram_similarities(camo_hist, bg_hist)
            
            bg_entropy = entropy(bg_hist + 1e-10)
            bg_uniformity = 1.0 - np.sum(bg_hist**2)
            bg_mean_brightness = np.sum(np.arange(256) * bg_hist)
            
            result['bg_histogram_properties'] = {
                'entropy': float(bg_entropy),
                'uniformity': float(bg_uniformity),
                'mean_brightness': float(bg_mean_brightness)
            }
            
            result['histogram_similarities'] = similarities
            
            # Overall histogram matching score
            hist_similarity_score = np.mean(list(similarities.values())) * 100
            result['histogram_matching_score'] = float(hist_similarity_score)
        
        return result
    
    def _calculate_histogram_similarities(self, hist1: np.ndarray, 
                                         hist2: np.ndarray) -> Dict[str, float]:
        """Calculate multiple histogram similarity measures"""
        
        similarities = {}
        
        # Correlation coefficient
        try:
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            if not np.isnan(correlation):
                similarities['correlation'] = float(correlation)
            else:
                similarities['correlation'] = 0.0
        except:
            similarities['correlation'] = 0.0
        
        # Chi-squared distance
        chi_squared = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
        similarities['chi_squared'] = float(max(0, 1 - chi_squared / 2))
        
        # Bhattacharyya coefficient
        bhattacharyya = np.sum(np.sqrt(hist1 * hist2))
        similarities['bhattacharyya'] = float(bhattacharyya)
        
        # Intersection
        intersection = np.sum(np.minimum(hist1, hist2))
        similarities['intersection'] = float(intersection)
        
        # Kullback-Leibler divergence
        try:
            kl_div = np.sum(hist1 * np.log((hist1 + 1e-10) / (hist2 + 1e-10)))
            similarities['kl_divergence'] = float(max(0, 1 - abs(kl_div)))
        except:
            similarities['kl_divergence'] = 0.0
        
        return similarities
    
    def _analyze_adaptive_brightness(self, camo_luminance: np.ndarray,
                                    bg_luminance: np.ndarray = None,
                                    options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze adaptive brightness properties"""
        
        # Local brightness adaptation
        adaptation_scales = [16, 32, 64]
        adaptation_results = {}
        
        for scale in adaptation_scales:
            # Calculate local brightness statistics
            local_stats = self._calculate_local_brightness_stats(camo_luminance, scale)
            adaptation_results[f'scale_{scale}'] = local_stats
        
        # Cross-scale consistency
        mean_values = [stats['local_brightness_variance'] for stats in adaptation_results.values()]
        consistency_score = 1.0 - (np.std(mean_values) / (np.mean(mean_values) + 1e-6))
        
        result = {
            'local_adaptation': adaptation_results,
            'cross_scale_consistency': float(consistency_score * 100)
        }
        
        # If background provided, analyze adaptation similarity
        if bg_luminance is not None:
            bg_adaptation_results = {}
            
            for scale in adaptation_scales:
                bg_local_stats = self._calculate_local_brightness_stats(bg_luminance, scale)
                bg_adaptation_results[f'scale_{scale}'] = bg_local_stats
            
            # Compare adaptation patterns
            adaptation_similarities = []
            for scale in adaptation_scales:
                camo_var = adaptation_results[f'scale_{scale}']['local_brightness_variance']
                bg_var = bg_adaptation_results[f'scale_{scale}']['local_brightness_variance']
                
                # Similarity based on variance ratio
                similarity = min(camo_var, bg_var) / max(camo_var, bg_var)
                adaptation_similarities.append(similarity)
            
            result['bg_local_adaptation'] = bg_adaptation_results
            result['adaptation_similarity_score'] = float(np.mean(adaptation_similarities) * 100)
        
        return result
    
    def _calculate_local_brightness_stats(self, luminance: np.ndarray, 
                                         window_size: int) -> Dict[str, Any]:
        """Calculate local brightness statistics"""
        
        # Calculate local mean and variance using convolution
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
        local_mean = cv2.filter2D(luminance.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((luminance.astype(np.float32) - local_mean)**2, -1, kernel)
        
        return {
            'local_brightness_mean': float(np.mean(local_mean)),
            'local_brightness_variance': float(np.mean(local_variance)),
            'local_brightness_uniformity': float(1.0 - (np.std(local_mean) / 255.0))
        }
    
    def _calculate_brightness_blend_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall brightness matching score"""
        
        scores = []
        weights = []
        
        # Basic brightness distribution score
        brightness_analysis = analysis_results.get('brightness_analysis', {})
        if 'brightness_matching_score' in brightness_analysis:
            scores.append(brightness_analysis['brightness_matching_score'])
            weights.append(0.25)
        
        if 'distribution_matching_score' in brightness_analysis:
            scores.append(brightness_analysis['distribution_matching_score'])
            weights.append(0.15)
        
        # Contrast analysis score
        contrast_analysis = analysis_results.get('contrast_analysis', {})
        if 'contrast_similarity_score' in contrast_analysis:
            scores.append(contrast_analysis['contrast_similarity_score'])
            weights.append(0.20)
        
        # Shadow analysis score
        shadow_analysis = analysis_results.get('shadow_analysis', {})
        if 'shadow_effectiveness_score' in shadow_analysis:
            scores.append(shadow_analysis['shadow_effectiveness_score'])
            weights.append(0.15)
        
        # Illumination adaptability score
        illumination_analysis = analysis_results.get('illumination_analysis', {})
        if 'overall_adaptability_score' in illumination_analysis:
            scores.append(illumination_analysis['overall_adaptability_score'])
            weights.append(0.15)
        
        # Histogram matching score
        histogram_analysis = analysis_results.get('histogram_analysis', {})
        if 'histogram_matching_score' in histogram_analysis:
            scores.append(histogram_analysis['histogram_matching_score'])
            weights.append(0.10)
        
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
            'pipeline_name': 'brightness_matching',
            'error': error_message,
            'score': 0.0,
            'execution_time': 0.0
        }