import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import math
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.metrics import auc

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('distance_simulator')


class DistanceDetectionSimulator:
    """
    Advanced distance-based detection simulation for camouflage effectiveness.
    Models how camouflage performance changes with viewing distance through
    atmospheric effects, resolution changes, and visual acuity limitations.
    """
    
    def __init__(self):
        self.standard_distances = config.get_standard_distances()
        self.analysis_params = config.get_analysis_params()
        
        # Physical and optical parameters
        self.human_visual_params = {
            'visual_angle_threshold': 1.0,  # arcminutes for detection
            'contrast_threshold': 0.02,     # Weber contrast threshold
            'spatial_frequency_limit': 60,  # cycles per degree
            'eye_resolution': 20/20         # Normal vision
        }
        
        # Atmospheric parameters
        self.atmospheric_params = {
            'visibility': 10000,  # meters (clear day)
            'scattering_coefficient': 0.01,
            'humidity': 0.5,
            'temperature': 20.0  # Celsius
        }
        
        logger.info(f"DistanceSimulator initialized with {len(self.standard_distances)} test distances")
    
    def simulate_distance_detection(self, camo_img: np.ndarray, 
                                   options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simulate detection probability across different viewing distances.
        
        Args:
            camo_img: Camouflage image in BGR format
            options: Simulation options
            
        Returns:
            Distance detection analysis results
        """
        start_time = cv2.getTickCount()
        logger.info("Starting distance-based detection simulation")
        
        if options is None:
            options = {}
        
        try:
            # Set object specifications if provided
            if 'object_size_meters' in options and 'reference_distance' in options:
                self.set_object_specifications(
                    options['object_size_meters'], 
                    options['reference_distance']
                )
            else:
                # Default: assume 1.8m person at 10m reference distance
                self.set_object_specifications(1.8, 10.0)
            
            # Step 1: Analyze base image characteristics
            logger.debug("Step 1: Analyzing base image characteristics")
            base_characteristics = self._analyze_base_characteristics(camo_img)
            
            # Step 2: Simulate at each test distance
            logger.debug("Step 2: Simulating detection at multiple distances")
            distance_results = {}
            
            for distance in self.standard_distances:
                logger.debug(f"Simulating at {distance}m")
                distance_result = self._simulate_at_distance(camo_img, distance, options)
                distance_results[f"{distance}m"] = distance_result
            
            # Step 3: Calculate detection probability curve
            logger.debug("Step 3: Calculating detection probability curve")
            detection_curve = self._generate_detection_curve(distance_results)
            
            # Step 4: Find critical detection distance
            logger.debug("Step 4: Finding critical detection distance")
            critical_distance = self._find_critical_detection_distance(detection_curve)
            
            # Step 5: Analyze distance effectiveness
            logger.debug("Step 5: Analyzing distance effectiveness")
            effectiveness_analysis = self._analyze_distance_effectiveness(
                distance_results, detection_curve, critical_distance
            )
            
            # Step 6: Calculate overall distance score
            distance_score = self._calculate_distance_effectiveness_score(
                effectiveness_analysis, critical_distance, detection_curve
            )
            
            execution_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            results = {
                'pipeline_name': 'distance_detection',
                'score': distance_score,
                'base_characteristics': base_characteristics,
                'distance_results': distance_results,
                'detection_curve': detection_curve,
                'critical_distance': critical_distance,
                'effectiveness_analysis': effectiveness_analysis,
                'execution_time': execution_time,
                'metadata': {
                    'test_distances': self.standard_distances,
                    'object_specifications': {
                        'size_meters': getattr(self, 'object_size_m', 1.8),
                        'reference_distance': getattr(self, 'reference_distance_m', 10.0)
                    },
                    'simulation_parameters': {
                        'atmospheric_visibility': self.atmospheric_params['visibility'],
                        'visual_acuity': self.human_visual_params['eye_resolution']
                    }
                }
            }
            
            logger.info(f"Distance simulation completed: score={distance_score:.1f}/100, critical_distance={critical_distance:.1f}m in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Distance simulation failed: {str(e)}")
            return self._create_error_result(str(e))
    
    def set_object_specifications(self, actual_size_m: float, ref_distance_m: float):
        """
        Set real-world object specifications for accurate distance modeling.
        
        Args:
            actual_size_m: Actual object size in meters
            ref_distance_m: Reference distance in meters
        """
        self.object_size_m = actual_size_m
        self.reference_distance_m = ref_distance_m
        
        logger.debug(f"Object specs set: {actual_size_m}m object at {ref_distance_m}m reference")
    
    def _analyze_base_characteristics(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze baseline image characteristics for distance modeling"""
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate baseline characteristics
        characteristics = {
            'resolution': {'width': img.shape[1], 'height': img.shape[0]},
            'contrast_measures': self._calculate_contrast_measures(gray),
            'edge_density': self._calculate_edge_density(gray),
            'texture_energy': self._calculate_texture_energy(gray),
            'frequency_content': self._analyze_frequency_content(gray),
            'noise_levels': self._estimate_noise_levels(gray)
        }
        
        return characteristics
    
    def _calculate_contrast_measures(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate multiple contrast measures"""
        
        # RMS contrast
        rms_contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        
        # Michelson contrast
        max_val = np.max(gray)
        min_val = np.min(gray)
        michelson_contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
        
        # Local contrast (using Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        local_contrast = np.std(laplacian)
        
        return {
            'rms_contrast': float(rms_contrast),
            'michelson_contrast': float(michelson_contrast),
            'local_contrast': float(local_contrast / 255.0)  # Normalize
        }
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density in image"""
        edges = cv2.Canny(gray, 50, 150)
        return float(np.sum(edges > 0) / edges.size)
    
    def _calculate_texture_energy(self, gray: np.ndarray) -> float:
        """Calculate texture energy using gradient magnitude"""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.mean(gradient_magnitude) / 255.0)
    
    def _analyze_frequency_content(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze spatial frequency content"""
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        magnitude = np.abs(np.fft.fftshift(fft))
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create frequency coordinate grids
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Define frequency bands
        max_freq = min(h, w) // 2
        low_freq = distances <= max_freq * 0.2
        mid_freq = (distances > max_freq * 0.2) & (distances <= max_freq * 0.6)
        high_freq = distances > max_freq * 0.6
        
        # Calculate energy in each band
        total_energy = np.sum(magnitude**2)
        if total_energy > 0:
            low_energy = np.sum((magnitude * low_freq)**2) / total_energy
            mid_energy = np.sum((magnitude * mid_freq)**2) / total_energy
            high_energy = np.sum((magnitude * high_freq)**2) / total_energy
        else:
            low_energy = mid_energy = high_energy = 0.0
        
        return {
            'low_frequency_energy': float(low_energy),
            'mid_frequency_energy': float(mid_energy),
            'high_frequency_energy': float(high_energy)
        }
    
    def _estimate_noise_levels(self, gray: np.ndarray) -> Dict[str, float]:
        """Estimate noise levels in image"""
        
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_estimate = np.var(laplacian)
        
        # Signal-to-noise ratio estimate
        signal_power = np.var(gray.astype(np.float64))
        snr = signal_power / noise_estimate if noise_estimate > 0 else float('inf')
        
        return {
            'noise_variance': float(noise_estimate),
            'estimated_snr': float(min(snr, 1000.0))  # Cap very high values
        }
    
    def _simulate_at_distance(self, img: np.ndarray, distance: float, 
                             options: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate image appearance and detection at specific distance"""
        
        # Calculate angular size at this distance
        angular_size = self.calculate_angular_size(self.object_size_m, distance)
        
        # Apply distance effects in sequence
        distance_img = img.copy()
        
        # 1. Resolution reduction based on distance
        distance_img = self.reduce_resolution_by_distance(distance_img, distance)
        
        # 2. Atmospheric effects
        if options.get('simulate_atmospheric_effects', True):
            distance_img = self.simulate_atmospheric_blur(distance_img, distance)
        
        # 3. Motion blur (if observer/object moving)
        if options.get('simulate_motion_blur', False):
            distance_img = self.apply_motion_blur(distance_img, distance)
        
        # Calculate detection metrics for this distance
        detection_probability = self.calculate_detection_probability(distance_img, distance)
        
        # Analyze image quality degradation
        quality_metrics = self._analyze_quality_degradation(img, distance_img)
        
        return {
            'distance_meters': distance,
            'angular_size_arcminutes': angular_size,
            'detection_probability': detection_probability,
            'simulated_image_shape': distance_img.shape,
            'quality_metrics': quality_metrics,
            'visibility_factors': self._calculate_visibility_factors(distance)
        }
    
    def calculate_angular_size(self, size_m: float, distance_m: float) -> float:
        """
        Calculate angular size in arcminutes.
        
        Args:
            size_m: Object size in meters
            distance_m: Distance in meters
            
        Returns:
            Angular size in arcminutes
        """
        angular_size_radians = 2 * math.atan(size_m / (2 * distance_m))
        angular_size_degrees = math.degrees(angular_size_radians)
        angular_size_arcminutes = angular_size_degrees * 60
        
        return angular_size_arcminutes
    
    def reduce_resolution_by_distance(self, img: np.ndarray, distance_m: float) -> np.ndarray:
        """
        Reduce image resolution based on distance and visual acuity.
        
        Args:
            img: Input image
            distance_m: Distance in meters
            
        Returns:
            Resolution-reduced image
        """
        # Calculate scale factor based on distance
        # Objects appear smaller and less detailed at greater distances
        reference_distance = getattr(self, 'reference_distance_m', 10.0)
        scale_factor = reference_distance / distance_m
        
        # Minimum scale factor to avoid tiny images
        scale_factor = max(0.1, min(1.0, scale_factor))
        
        h, w = img.shape[:2]
        new_h = max(10, int(h * scale_factor))
        new_w = max(10, int(w * scale_factor))
        
        # Resize with appropriate interpolation
        if scale_factor < 1.0:
            # Downsampling - use area interpolation to avoid aliasing
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # No scaling needed
            resized = img.copy()
        
        # Resize back to original size to maintain consistency
        final_img = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return final_img
    
    def simulate_atmospheric_blur(self, img: np.ndarray, distance_m: float) -> np.ndarray:
        """
        Apply atmospheric scattering and blur effects.
        
        Args:
            img: Input image
            distance_m: Distance in meters
            
        Returns:
            Atmospherically blurred image
        """
        # Calculate atmospheric effects
        visibility = self.atmospheric_params['visibility']
        
        # Extinction coefficient (how much light is scattered/absorbed)
        extinction = distance_m / visibility
        
        # Blur kernel size increases with distance
        blur_sigma = max(0.5, extinction * 2.0)
        kernel_size = int(blur_sigma * 6) | 1  # Ensure odd size
        kernel_size = min(kernel_size, 15)  # Cap maximum blur
        
        # Apply Gaussian blur for atmospheric scattering
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), blur_sigma)
        
        # Atmospheric haze (reduces contrast)
        haze_factor = min(0.3, extinction * 0.1)
        
        # Add atmospheric haze
        hazy_img = blurred.astype(np.float32)
        mean_brightness = np.mean(hazy_img)
        hazy_img = hazy_img * (1 - haze_factor) + mean_brightness * haze_factor
        
        # Ensure valid range
        hazy_img = np.clip(hazy_img, 0, 255).astype(np.uint8)
        
        return hazy_img
    
    def apply_motion_blur(self, img: np.ndarray, distance_m: float) -> np.ndarray:
        """
        Apply motion blur due to observer/object movement.
        
        Args:
            img: Input image
            distance_m: Distance in meters
            
        Returns:
            Motion-blurred image
        """
        # Motion blur is more noticeable at closer distances
        # and for faster relative motion
        
        # Assume some relative motion (walking speed ~1.4 m/s, observation time ~0.1s)
        relative_speed = 1.4  # m/s
        observation_time = 0.1  # seconds
        
        # Calculate angular motion
        linear_motion = relative_speed * observation_time
        angular_motion_rad = linear_motion / distance_m
        
        # Convert to pixel motion (rough estimate)
        image_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
        pixel_motion = angular_motion_rad * image_diagonal * 0.1  # Scale factor
        
        motion_length = max(1, int(pixel_motion))
        motion_length = min(motion_length, 15)  # Cap motion blur
        
        if motion_length > 1:
            # Create motion blur kernel (horizontal motion)
            kernel = np.zeros((motion_length, motion_length))
            kernel[motion_length // 2, :] = 1.0 / motion_length
            
            # Apply motion blur
            blurred = cv2.filter2D(img, -1, kernel)
            return blurred
        
        return img
    
    def calculate_detection_probability(self, img: np.ndarray, distance_m: float) -> float:
        """
        Calculate probability of object detection at given distance.
        
        Args:
            img: Image as seen at distance
            distance_m: Distance in meters
            
        Returns:
            Detection probability (0.0 to 1.0)
        """
        # Multiple factors contribute to detection probability
        
        # 1. Angular size factor
        angular_size = self.calculate_angular_size(self.object_size_m, distance_m)
        angular_threshold = self.human_visual_params['visual_angle_threshold']
        
        if angular_size > angular_threshold:
            angular_factor = 1.0
        else:
            angular_factor = angular_size / angular_threshold
        
        # 2. Contrast factor
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_contrast = np.std(gray) / (np.mean(gray) + 1e-6)
        contrast_threshold = self.human_visual_params['contrast_threshold']
        
        if image_contrast > contrast_threshold:
            contrast_factor = 1.0
        else:
            contrast_factor = image_contrast / contrast_threshold
        
        # 3. Resolution factor (based on remaining detail)
        edge_density = self._calculate_edge_density(gray)
        resolution_factor = min(1.0, edge_density * 10)  # Scale factor
        
        # 4. Atmospheric visibility factor
        visibility = self.atmospheric_params['visibility']
        visibility_factor = max(0.1, min(1.0, visibility / distance_m))
        
        # Combine factors (multiplicative model)
        detection_prob = angular_factor * contrast_factor * resolution_factor * visibility_factor
        
        # Apply psychophysical detection curve
        # Human detection follows a sigmoid curve
        detection_prob = 1 / (1 + np.exp(-5 * (detection_prob - 0.5)))
        
        return max(0.0, min(1.0, detection_prob))
    
    def _analyze_quality_degradation(self, original: np.ndarray, 
                                   degraded: np.ndarray) -> Dict[str, float]:
        """Analyze image quality degradation"""
        
        # Ensure same size for comparison
        if degraded.shape != original.shape:
            degraded = cv2.resize(degraded, (original.shape[1], original.shape[0]))
        
        # Convert to grayscale for analysis
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        deg_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        
        # Peak Signal-to-Noise Ratio
        mse = np.mean((orig_gray.astype(np.float32) - deg_gray.astype(np.float32))**2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Structural Similarity Index (simplified)
        mean_orig = np.mean(orig_gray)
        mean_deg = np.mean(deg_gray)
        var_orig = np.var(orig_gray)
        var_deg = np.var(deg_gray)
        cov = np.mean((orig_gray - mean_orig) * (deg_gray - mean_deg))
        
        ssim_numerator = (2 * mean_orig * mean_deg + 1) * (2 * cov + 1)
        ssim_denominator = (mean_orig**2 + mean_deg**2 + 1) * (var_orig + var_deg + 1)
        ssim = ssim_numerator / ssim_denominator if ssim_denominator > 0 else 0
        
        # Detail preservation (correlation)
        try:
            correlation = np.corrcoef(orig_gray.flatten(), deg_gray.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        return {
            'psnr': float(min(psnr, 100.0)),  # Cap very high PSNR values
            'ssim': float(max(0, ssim)),
            'correlation': float(max(0, correlation)),
            'quality_score': float((ssim + max(0, correlation)) / 2 * 100)
        }
    
    def _calculate_visibility_factors(self, distance_m: float) -> Dict[str, float]:
        """Calculate various visibility factors at distance"""
        
        visibility_range = self.atmospheric_params['visibility']
        
        return {
            'atmospheric_transmission': float(max(0.1, 1.0 - distance_m / visibility_range)),
            'angular_size_factor': float(self.reference_distance_m / distance_m),
            'diffraction_limit': float(max(0.1, 1.0 - distance_m / 1000.0)),  # Simplified
            'scattering_factor': float(max(0.1, 1.0 - distance_m * self.atmospheric_params['scattering_coefficient']))
        }
    
    def _generate_detection_curve(self, distance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detection probability vs distance curve"""
        
        distances = []
        probabilities = []
        
        # Extract data points
        for key, result in distance_results.items():
            if 'detection_probability' in result:
                distance = result['distance_meters']
                probability = result['detection_probability']
                distances.append(distance)
                probabilities.append(probability)
        
        if len(distances) < 2:
            return {'distances': [], 'detection_probabilities': [], 'curve_fit': None}
        
        # Sort by distance
        sorted_pairs = sorted(zip(distances, probabilities))
        distances, probabilities = zip(*sorted_pairs)
        
        # Fit curve (exponential decay model)
        try:
            # Simple exponential fit: P = a * exp(-b * d) + c
            from scipy.optimize import curve_fit
            
            def exp_decay(d, a, b, c):
                return a * np.exp(-b * d) + c
            
            popt, _ = curve_fit(exp_decay, distances, probabilities, 
                              bounds=([0, 0, 0], [2, 1, 1]),
                              maxfev=1000)
            
            curve_fit_params = {'a': float(popt[0]), 'b': float(popt[1]), 'c': float(popt[2])}
            
            # Generate smooth curve
            smooth_distances = np.linspace(min(distances), max(distances), 100)
            smooth_probabilities = exp_decay(smooth_distances, *popt)
            
        except:
            # Fallback to linear interpolation
            curve_fit_params = None
            smooth_distances = distances
            smooth_probabilities = probabilities
        
        return {
            'distances': list(distances),
            'detection_probabilities': list(probabilities),
            'smooth_curve': {
                'distances': list(smooth_distances),
                'probabilities': list(smooth_probabilities)
            },
            'curve_fit': curve_fit_params
        }
    
    def _find_critical_detection_distance(self, detection_curve: Dict[str, Any]) -> float:
        """Find distance where detection probability = 50%"""
        
        distances = detection_curve.get('distances', [])
        probabilities = detection_curve.get('detection_probabilities', [])
        
        if len(distances) < 2:
            return 0.0
        
        try:
            # Interpolate to find 50% detection distance
            if min(probabilities) <= 0.5 <= max(probabilities):
                interp_func = interp1d(probabilities, distances, kind='linear')
                critical_distance = float(interp_func(0.5))
            else:
                # Extrapolate if 50% not in range
                if min(probabilities) > 0.5:
                    # All probabilities above 50% - critical distance is beyond max distance
                    critical_distance = max(distances) * 1.5
                else:
                    # All probabilities below 50% - critical distance is before min distance
                    critical_distance = min(distances) * 0.5
            
        except:
            # Fallback: estimate from available data
            closest_idx = np.argmin([abs(p - 0.5) for p in probabilities])
            critical_distance = distances[closest_idx]
        
        return max(0.0, critical_distance)
    
    def _analyze_distance_effectiveness(self, distance_results: Dict[str, Any],
                                      detection_curve: Dict[str, Any],
                                      critical_distance: float) -> Dict[str, Any]:
        """Analyze overall distance effectiveness"""
        
        # Extract distances and probabilities
        distances = detection_curve.get('distances', [])
        probabilities = detection_curve.get('detection_probabilities', [])
        
        if len(distances) < 2:
            return {'effectiveness_score': 0.0}
        
        # Calculate area under the curve (lower is better for camouflage)
        try:
            curve_area = auc(distances, probabilities)
            max_area = max(distances) - min(distances)  # Maximum possible area
            normalized_area = curve_area / max_area if max_area > 0 else 0
        except:
            normalized_area = np.mean(probabilities)
        
        # Distance range effectiveness
        max_distance = max(distances)
        effective_range = max_distance - critical_distance if critical_distance < max_distance else 0
        range_effectiveness = effective_range / max_distance if max_distance > 0 else 0
        
        # Steepness of detection falloff (steeper is better for camouflage)
        if len(probabilities) > 1:
            prob_gradient = np.gradient(probabilities, distances)
            avg_steepness = abs(np.mean(prob_gradient))
        else:
            avg_steepness = 0
        
        # Performance at standard distances
        standard_performance = {}
        for distance in [25, 50, 100]:  # Key tactical distances
            if distance in distances:
                idx = distances.index(distance)
                standard_performance[f'{distance}m'] = probabilities[idx]
        
        return {
            'curve_area': float(curve_area) if 'curve_area' in locals() else 0.0,
            'normalized_area': float(normalized_area),
            'effective_range': float(effective_range),
            'range_effectiveness': float(range_effectiveness * 100),
            'detection_falloff_rate': float(avg_steepness * 100),
            'standard_distance_performance': standard_performance
        }
    
    def _calculate_distance_effectiveness_score(self, effectiveness_analysis: Dict[str, Any],
                                               critical_distance: float,
                                               detection_curve: Dict[str, Any]) -> float:
        """Calculate overall distance effectiveness score"""
        
        scores = []
        weights = []
        
        # Critical distance score (longer is better)
        max_test_distance = max(self.standard_distances)
        critical_distance_score = min(100, (critical_distance / max_test_distance) * 100)
        scores.append(critical_distance_score)
        weights.append(0.3)
        
        # Range effectiveness score
        range_effectiveness = effectiveness_analysis.get('range_effectiveness', 0)
        scores.append(range_effectiveness)
        weights.append(0.25)
        
        # Detection falloff rate (steeper is better)
        falloff_rate = effectiveness_analysis.get('detection_falloff_rate', 0)
        falloff_score = min(100, falloff_rate * 10)  # Scale factor
        scores.append(falloff_score)
        weights.append(0.2)
        
        # Low detection at long distances
        probabilities = detection_curve.get('detection_probabilities', [])
        if probabilities:
            long_distance_performance = 100 * (1 - probabilities[-1])  # Last (longest) distance
            scores.append(long_distance_performance)
            weights.append(0.25)
        
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
            'pipeline_name': 'distance_detection',
            'error': error_message,
            'score': 0.0,
            'execution_time': 0.0
        }