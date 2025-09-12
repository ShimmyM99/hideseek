import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.metrics import pairwise_distances
import math

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('pattern_analyzer')


class PatternDisruptionAnalyzer:
    """
    Advanced pattern disruption and feature detection analysis for camouflage effectiveness.
    Analyzes how well patterns break up recognizable shapes and disrupt object detection.
    """
    
    def __init__(self):
        self.feature_detectors = config.get('analysis.feature_detector', 'ORB')
        self.texture_window = config.get('analysis.texture_window_size', 32)
        self.edge_detector = config.get('analysis.edge_detector', 'canny')
        
        # Initialize feature detectors
        self._init_feature_detectors()
        
        # Pattern analysis parameters
        self.gabor_params = {
            'frequencies': [0.1, 0.2, 0.3, 0.4],
            'orientations': [0, 45, 90, 135],
            'sigma_x': 2.0,
            'sigma_y': 2.0
        }
        
        logger.info(f"PatternAnalyzer initialized with {self.feature_detectors} detector")
    
    def _init_feature_detectors(self):
        """Initialize OpenCV feature detectors"""
        self.detectors = {}
        
        # ORB detector
        try:
            self.detectors['ORB'] = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        except Exception as e:
            logger.warning(f"Failed to initialize ORB: {e}")
        
        # SIFT detector (if available)
        try:
            self.detectors['SIFT'] = cv2.SIFT_create(nfeatures=1000)
        except Exception as e:
            logger.warning(f"SIFT not available: {e}")
        
        # SURF detector (if available)
        try:
            self.detectors['SURF'] = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        except Exception as e:
            logger.warning(f"SURF not available: {e}")
        
        # BRISK detector
        try:
            self.detectors['BRISK'] = cv2.BRISK_create()
        except Exception as e:
            logger.warning(f"Failed to initialize BRISK: {e}")
        
        logger.info(f"Available feature detectors: {list(self.detectors.keys())}")
    
    def analyze_pattern_disruption(self, camo_img: np.ndarray, bg_img: np.ndarray = None,
                                  options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive pattern disruption analysis.
        
        Args:
            camo_img: Camouflage image in BGR format
            bg_img: Optional background reference image
            options: Analysis options
            
        Returns:
            Pattern disruption analysis results
        """
        start_time = cv2.getTickCount()
        logger.info("Starting pattern disruption analysis")
        
        if options is None:
            options = {}
        
        try:
            # Convert to grayscale for feature analysis
            camo_gray = cv2.cvtColor(camo_img, cv2.COLOR_BGR2GRAY)
            bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY) if bg_img is not None else None
            
            # Step 1: Feature Detection and Extraction
            logger.debug("Step 1: Feature detection")
            feature_analysis = self._analyze_features(camo_gray, bg_gray, options)
            
            # Step 2: Edge Analysis and Continuity
            logger.debug("Step 2: Edge continuity analysis")
            edge_analysis = self._analyze_edge_continuity(camo_gray, camo_img, options)
            
            # Step 3: Texture Analysis
            logger.debug("Step 3: Texture similarity analysis")
            texture_analysis = self._analyze_texture_similarity(camo_gray, bg_gray, options)
            
            # Step 4: Pattern Complexity Analysis
            logger.debug("Step 4: Pattern complexity analysis")
            complexity_analysis = self._analyze_pattern_complexity(camo_gray, options)
            
            # Step 5: Shape Breakup Analysis
            logger.debug("Step 5: Shape breakup analysis")
            shape_analysis = self._analyze_shape_breakup(camo_gray, camo_img, options)
            
            # Step 6: Spatial Frequency Analysis
            logger.debug("Step 6: Spatial frequency analysis")
            frequency_analysis = self._analyze_spatial_frequencies(camo_gray, options)
            
            # Step 7: Calculate overall pattern disruption score
            pattern_score = self._calculate_pattern_disruption_score({
                'feature_analysis': feature_analysis,
                'edge_analysis': edge_analysis,
                'texture_analysis': texture_analysis,
                'complexity_analysis': complexity_analysis,
                'shape_analysis': shape_analysis,
                'frequency_analysis': frequency_analysis
            })
            
            execution_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            results = {
                'pipeline_name': 'pattern_disruption',
                'score': pattern_score,
                'feature_analysis': feature_analysis,
                'edge_analysis': edge_analysis,
                'texture_analysis': texture_analysis,
                'complexity_analysis': complexity_analysis,
                'shape_analysis': shape_analysis,
                'frequency_analysis': frequency_analysis,
                'execution_time': execution_time,
                'metadata': {
                    'detectors_used': list(self.detectors.keys()),
                    'texture_window_size': self.texture_window,
                    'edge_detector': self.edge_detector
                }
            }
            
            logger.info(f"Pattern analysis completed: score={pattern_score:.1f}/100 in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")
            return self._create_error_result(str(e))
    
    def _analyze_features(self, camo_gray: np.ndarray, bg_gray: np.ndarray = None,
                         options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze feature detection and matching"""
        
        results = {}
        
        # Extract features using available detectors
        for detector_name, detector in self.detectors.items():
            try:
                logger.debug(f"Extracting features with {detector_name}")
                
                # Extract keypoints and descriptors
                kp_camo, desc_camo = detector.detectAndCompute(camo_gray, None)
                
                # Feature density analysis
                feature_density = len(kp_camo) / (camo_gray.shape[0] * camo_gray.shape[1])
                
                # Feature distribution analysis
                feature_distribution = self._analyze_feature_distribution(kp_camo, camo_gray.shape)
                
                # Feature strength analysis
                feature_strengths = [kp.response for kp in kp_camo] if hasattr(kp_camo[0], 'response') and len(kp_camo) > 0 else []
                
                detector_results = {
                    'feature_count': len(kp_camo),
                    'feature_density': feature_density,
                    'feature_distribution': feature_distribution,
                    'feature_strength_stats': {
                        'mean': np.mean(feature_strengths) if feature_strengths else 0,
                        'std': np.std(feature_strengths) if feature_strengths else 0,
                        'max': np.max(feature_strengths) if feature_strengths else 0
                    }
                }
                
                # If background provided, analyze feature matching
                if bg_gray is not None and desc_camo is not None:
                    try:
                        kp_bg, desc_bg = detector.detectAndCompute(bg_gray, None)
                        
                        if desc_bg is not None and len(desc_bg) > 0:
                            matching_analysis = self._analyze_feature_matching(
                                desc_camo, desc_bg, detector_name
                            )
                            detector_results['matching_analysis'] = matching_analysis
                    except Exception as e:
                        logger.warning(f"Feature matching failed for {detector_name}: {e}")
                
                results[detector_name] = detector_results
                
            except Exception as e:
                logger.warning(f"Feature extraction failed for {detector_name}: {e}")
                results[detector_name] = {'error': str(e)}
        
        return results
    
    def _analyze_feature_distribution(self, keypoints: List, img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze spatial distribution of detected features"""
        if not keypoints:
            return {'uniformity_score': 0.0, 'clustering_score': 0.0}
        
        # Extract keypoint coordinates
        coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        # Analyze spatial distribution uniformity
        h, w = img_shape
        
        # Divide image into grid and count features per cell
        grid_size = 8
        cell_h, cell_w = h // grid_size, w // grid_size
        grid_counts = np.zeros((grid_size, grid_size))
        
        for x, y in coords:
            grid_x = min(int(x // cell_w), grid_size - 1)
            grid_y = min(int(y // cell_h), grid_size - 1)
            grid_counts[grid_y, grid_x] += 1
        
        # Calculate uniformity (lower variance = more uniform)
        expected_count = len(keypoints) / (grid_size * grid_size)
        uniformity_variance = np.var(grid_counts)
        uniformity_score = max(0, 1 - uniformity_variance / (expected_count + 1))
        
        # Analyze clustering using pairwise distances
        if len(coords) > 1:
            distances = pdist(coords)
            clustering_score = 1.0 / (1.0 + np.std(distances) / np.mean(distances))
        else:
            clustering_score = 0.0
        
        return {
            'uniformity_score': float(uniformity_score),
            'clustering_score': float(clustering_score),
            'grid_distribution': grid_counts.tolist()
        }
    
    def _analyze_feature_matching(self, desc1: np.ndarray, desc2: np.ndarray, 
                                 detector_name: str) -> Dict[str, Any]:
        """Analyze feature matching between camouflage and background"""
        
        try:
            # Create matcher based on descriptor type
            if detector_name in ['ORB', 'BRISK']:
                # Binary descriptors
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                # Float descriptors (SIFT, SURF)
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            
            # Find matches using k-nearest neighbors
            matches = matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test (Lowe's ratio test)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Calculate matching statistics
            total_features = len(desc1)
            match_ratio = len(good_matches) / total_features if total_features > 0 else 0
            
            # Calculate match quality (lower distance = better match)
            if good_matches:
                distances = [m.distance for m in good_matches]
                avg_distance = np.mean(distances)
                distance_std = np.std(distances)
            else:
                avg_distance = float('inf')
                distance_std = 0
            
            # Pattern disruption score (higher match ratio = less disruption)
            disruption_score = max(0, 100 * (1 - match_ratio))
            
            return {
                'total_matches': len(good_matches),
                'match_ratio': float(match_ratio),
                'average_distance': float(avg_distance),
                'distance_std': float(distance_std),
                'disruption_score': float(disruption_score)
            }
            
        except Exception as e:
            logger.warning(f"Feature matching analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_edge_continuity(self, camo_gray: np.ndarray, camo_color: np.ndarray,
                               options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze edge continuity and disruption"""
        
        # Extract edges using multiple methods
        edge_maps = {}
        
        # Canny edge detection
        edges_canny = cv2.Canny(camo_gray, 50, 150)
        edge_maps['canny'] = edges_canny
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(camo_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(camo_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edges_sobel = (sobel_magnitude > np.mean(sobel_magnitude) + np.std(sobel_magnitude)).astype(np.uint8) * 255
        edge_maps['sobel'] = edges_sobel
        
        # Laplacian edge detection
        laplacian = cv2.Laplacian(camo_gray, cv2.CV_64F)
        edges_laplacian = (np.abs(laplacian) > np.mean(np.abs(laplacian)) + np.std(np.abs(laplacian))).astype(np.uint8) * 255
        edge_maps['laplacian'] = edges_laplacian
        
        results = {}
        
        for method, edge_map in edge_maps.items():
            # Edge density
            edge_density = np.sum(edge_map > 0) / edge_map.size
            
            # Edge continuity analysis
            continuity_score = self._measure_edge_continuity(edge_map)
            
            # Edge fragmentation analysis
            fragmentation_score = self._measure_edge_fragmentation(edge_map)
            
            results[method] = {
                'edge_density': float(edge_density),
                'continuity_score': float(continuity_score),
                'fragmentation_score': float(fragmentation_score)
            }
        
        # Overall edge disruption score
        avg_continuity = np.mean([r['continuity_score'] for r in results.values()])
        avg_fragmentation = np.mean([r['fragmentation_score'] for r in results.values()])
        
        # Lower continuity and higher fragmentation = better disruption
        edge_disruption_score = (100 - avg_continuity * 100) * 0.6 + avg_fragmentation * 100 * 0.4
        
        results['overall'] = {
            'edge_disruption_score': float(max(0, min(100, edge_disruption_score))),
            'average_continuity': float(avg_continuity),
            'average_fragmentation': float(avg_fragmentation)
        }
        
        return results
    
    def _measure_edge_continuity(self, edge_map: np.ndarray) -> float:
        """Measure edge continuity using connected components"""
        
        # Find connected components in edge map
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_map, connectivity=8)
        
        if num_labels <= 1:
            return 0.0
        
        # Analyze component sizes (excluding background)
        component_sizes = stats[1:, cv2.CC_STAT_AREA]  # Skip background (label 0)
        
        if len(component_sizes) == 0:
            return 0.0
        
        # Calculate continuity metrics
        total_edge_pixels = np.sum(edge_map > 0)
        if total_edge_pixels == 0:
            return 0.0
        
        # Longer edge segments indicate more continuity
        avg_component_size = np.mean(component_sizes)
        max_component_size = np.max(component_sizes)
        
        # Normalize by total edge pixels
        continuity_score = (avg_component_size + max_component_size * 0.5) / total_edge_pixels
        
        return min(1.0, continuity_score)
    
    def _measure_edge_fragmentation(self, edge_map: np.ndarray) -> float:
        """Measure edge fragmentation (how broken up edges are)"""
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_map, connectivity=8)
        
        if num_labels <= 1:
            return 0.0
        
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        total_edge_pixels = np.sum(edge_map > 0)
        
        if total_edge_pixels == 0:
            return 0.0
        
        # More components with smaller average size = higher fragmentation
        num_components = len(component_sizes)
        avg_component_size = np.mean(component_sizes)
        
        # Fragmentation score: more small components = higher score
        fragmentation_score = num_components / (avg_component_size + 1)
        
        # Normalize
        max_possible_fragmentation = total_edge_pixels  # Each pixel is its own component
        fragmentation_score = fragmentation_score / max_possible_fragmentation
        
        return min(1.0, fragmentation_score)
    
    def _analyze_texture_similarity(self, camo_gray: np.ndarray, bg_gray: np.ndarray = None,
                                   options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze texture similarity using multiple methods"""
        
        results = {}
        
        # Method 1: Gray-Level Co-occurrence Matrix (GLCM)
        results['glcm'] = self._analyze_glcm_texture(camo_gray, bg_gray)
        
        # Method 2: Local Binary Patterns (LBP)
        results['lbp'] = self._analyze_lbp_texture(camo_gray, bg_gray)
        
        # Method 3: Gabor Filter Bank
        results['gabor'] = self._analyze_gabor_texture(camo_gray, bg_gray)
        
        # Method 4: Statistical texture measures
        results['statistical'] = self._analyze_statistical_texture(camo_gray, bg_gray)
        
        # Overall texture similarity score
        similarity_scores = []
        for method, result in results.items():
            if 'similarity_score' in result:
                similarity_scores.append(result['similarity_score'])
        
        if similarity_scores:
            overall_similarity = np.mean(similarity_scores)
            # Higher similarity = better camouflage (for texture matching)
            texture_score = overall_similarity * 100
        else:
            texture_score = 0.0
        
        results['overall'] = {
            'texture_similarity_score': float(max(0, min(100, texture_score))),
            'methods_used': list(results.keys())
        }
        
        return results
    
    def _analyze_glcm_texture(self, camo_gray: np.ndarray, bg_gray: np.ndarray = None) -> Dict[str, Any]:
        """Analyze texture using Gray-Level Co-occurrence Matrix"""
        
        try:
            # Calculate GLCM for camouflage image
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            
            # Reduce gray levels for computational efficiency
            camo_reduced = (camo_gray // 32) * 32  # Reduce to 8 levels
            
            glcm_camo = graycomatrix(
                camo_reduced, distances=distances, angles=angles, 
                levels=256, symmetric=True, normed=True
            )
            
            # Extract texture features
            camo_features = {
                'contrast': graycoprops(glcm_camo, 'contrast').mean(),
                'dissimilarity': graycoprops(glcm_camo, 'dissimilarity').mean(),
                'homogeneity': graycoprops(glcm_camo, 'homogeneity').mean(),
                'energy': graycoprops(glcm_camo, 'energy').mean(),
                'correlation': graycoprops(glcm_camo, 'correlation').mean()
            }
            
            result = {'camo_features': camo_features}
            
            # If background provided, compare features
            if bg_gray is not None:
                try:
                    # Ensure same size
                    if bg_gray.shape != camo_gray.shape:
                        bg_gray = cv2.resize(bg_gray, (camo_gray.shape[1], camo_gray.shape[0]))
                    
                    bg_reduced = (bg_gray // 32) * 32
                    glcm_bg = graycomatrix(
                        bg_reduced, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True
                    )
                    
                    bg_features = {
                        'contrast': graycoprops(glcm_bg, 'contrast').mean(),
                        'dissimilarity': graycoprops(glcm_bg, 'dissimilarity').mean(),
                        'homogeneity': graycoprops(glcm_bg, 'homogeneity').mean(),
                        'energy': graycoprops(glcm_bg, 'energy').mean(),
                        'correlation': graycoprops(glcm_bg, 'correlation').mean()
                    }
                    
                    # Calculate similarity
                    feature_diffs = []
                    for key in camo_features:
                        if key in bg_features:
                            diff = abs(camo_features[key] - bg_features[key])
                            feature_diffs.append(diff)
                    
                    if feature_diffs:
                        avg_diff = np.mean(feature_diffs)
                        similarity = max(0, 1 - avg_diff)
                    else:
                        similarity = 0
                    
                    result['bg_features'] = bg_features
                    result['similarity_score'] = float(similarity)
                    
                except Exception as e:
                    logger.warning(f"GLCM background analysis failed: {e}")
            
            return result
            
        except Exception as e:
            logger.warning(f"GLCM analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_lbp_texture(self, camo_gray: np.ndarray, bg_gray: np.ndarray = None) -> Dict[str, Any]:
        """Analyze texture using Local Binary Patterns"""
        
        try:
            # LBP parameters
            radius = 3
            n_points = 8 * radius
            
            # Calculate LBP for camouflage
            lbp_camo = local_binary_pattern(camo_gray, n_points, radius, method='uniform')
            
            # Calculate LBP histogram
            hist_camo, _ = np.histogram(lbp_camo.ravel(), bins=n_points + 2, 
                                      range=(0, n_points + 2), density=True)
            
            result = {'camo_histogram': hist_camo.tolist()}
            
            # If background provided
            if bg_gray is not None:
                try:
                    if bg_gray.shape != camo_gray.shape:
                        bg_gray = cv2.resize(bg_gray, (camo_gray.shape[1], camo_gray.shape[0]))
                    
                    lbp_bg = local_binary_pattern(bg_gray, n_points, radius, method='uniform')
                    hist_bg, _ = np.histogram(lbp_bg.ravel(), bins=n_points + 2,
                                            range=(0, n_points + 2), density=True)
                    
                    # Calculate histogram similarity using chi-squared distance
                    chi_squared = np.sum((hist_camo - hist_bg)**2 / (hist_camo + hist_bg + 1e-10))
                    similarity = max(0, 1 - chi_squared / 2)
                    
                    result['bg_histogram'] = hist_bg.tolist()
                    result['similarity_score'] = float(similarity)
                    
                except Exception as e:
                    logger.warning(f"LBP background analysis failed: {e}")
            
            return result
            
        except Exception as e:
            logger.warning(f"LBP analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_gabor_texture(self, camo_gray: np.ndarray, bg_gray: np.ndarray = None) -> Dict[str, Any]:
        """Analyze texture using Gabor filter bank"""
        
        try:
            # Generate Gabor filter responses
            camo_responses = self._apply_gabor_filters(camo_gray)
            
            # Calculate statistical measures from responses
            camo_features = self._extract_gabor_features(camo_responses)
            
            result = {'camo_features': camo_features}
            
            # If background provided
            if bg_gray is not None:
                try:
                    if bg_gray.shape != camo_gray.shape:
                        bg_gray = cv2.resize(bg_gray, (camo_gray.shape[1], camo_gray.shape[0]))
                    
                    bg_responses = self._apply_gabor_filters(bg_gray)
                    bg_features = self._extract_gabor_features(bg_responses)
                    
                    # Calculate feature similarity
                    similarities = []
                    for key in camo_features:
                        if key in bg_features:
                            camo_vals = np.array(camo_features[key])
                            bg_vals = np.array(bg_features[key])
                            
                            # Normalize and calculate correlation
                            if np.std(camo_vals) > 0 and np.std(bg_vals) > 0:
                                corr = np.corrcoef(camo_vals, bg_vals)[0, 1]
                                if not np.isnan(corr):
                                    similarities.append(abs(corr))
                    
                    similarity = np.mean(similarities) if similarities else 0
                    
                    result['bg_features'] = bg_features
                    result['similarity_score'] = float(similarity)
                    
                except Exception as e:
                    logger.warning(f"Gabor background analysis failed: {e}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Gabor analysis failed: {e}")
            return {'error': str(e)}
    
    def _apply_gabor_filters(self, img: np.ndarray) -> List[np.ndarray]:
        """Apply bank of Gabor filters to image"""
        
        responses = []
        
        for freq in self.gabor_params['frequencies']:
            for theta in self.gabor_params['orientations']:
                # Convert angle to radians
                theta_rad = np.deg2rad(theta)
                
                # Create Gabor kernel
                kernel_size = 31
                sigma_x = self.gabor_params['sigma_x']
                sigma_y = self.gabor_params['sigma_y']
                
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size), sigma_x, theta_rad, 
                    2 * np.pi / freq, 0.5, 0, ktype=cv2.CV_32F
                )
                
                # Apply filter
                response = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                responses.append(response)
        
        return responses
    
    def _extract_gabor_features(self, responses: List[np.ndarray]) -> Dict[str, List[float]]:
        """Extract statistical features from Gabor filter responses"""
        
        features = {
            'means': [],
            'stds': [],
            'energies': [],
            'entropies': []
        }
        
        for response in responses:
            # Statistical measures
            features['means'].append(float(np.mean(response)))
            features['stds'].append(float(np.std(response)))
            features['energies'].append(float(np.sum(response**2)))
            
            # Entropy calculation
            hist, _ = np.histogram(response.ravel(), bins=32, density=True)
            hist = hist + 1e-10  # Avoid log(0)
            features['entropies'].append(float(entropy(hist)))
        
        return features
    
    def _analyze_statistical_texture(self, camo_gray: np.ndarray, 
                                   bg_gray: np.ndarray = None) -> Dict[str, Any]:
        """Analyze texture using statistical measures"""
        
        def extract_statistical_features(img):
            # First-order statistics
            mean = np.mean(img)
            std = np.std(img)
            skewness = float(ndimage.measurements.standard_deviation(img))
            
            # Second-order statistics using gradients
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            return {
                'mean': float(mean),
                'std': float(std),
                'gradient_mean': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude)),
                'direction_uniformity': float(np.std(gradient_direction))
            }
        
        # Extract features for camouflage
        camo_features = extract_statistical_features(camo_gray)
        
        result = {'camo_features': camo_features}
        
        # If background provided
        if bg_gray is not None:
            try:
                if bg_gray.shape != camo_gray.shape:
                    bg_gray = cv2.resize(bg_gray, (camo_gray.shape[1], camo_gray.shape[0]))
                
                bg_features = extract_statistical_features(bg_gray)
                
                # Calculate similarity
                similarities = []
                for key in camo_features:
                    if key in bg_features:
                        diff = abs(camo_features[key] - bg_features[key])
                        # Normalize by the larger value
                        max_val = max(abs(camo_features[key]), abs(bg_features[key]), 1)
                        similarity = 1 - (diff / max_val)
                        similarities.append(max(0, similarity))
                
                overall_similarity = np.mean(similarities) if similarities else 0
                
                result['bg_features'] = bg_features
                result['similarity_score'] = float(overall_similarity)
                
            except Exception as e:
                logger.warning(f"Statistical background analysis failed: {e}")
        
        return result
    
    def _analyze_pattern_complexity(self, camo_gray: np.ndarray, 
                                   options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pattern complexity using multiple measures"""
        
        # Fractal dimension estimation
        fractal_dim = self._estimate_fractal_dimension(camo_gray)
        
        # Entropy-based complexity
        entropy_complexity = self._calculate_entropy_complexity(camo_gray)
        
        # Fourier-based complexity
        fourier_complexity = self._calculate_fourier_complexity(camo_gray)
        
        # Overall complexity score
        complexity_score = (fractal_dim + entropy_complexity + fourier_complexity) / 3 * 100
        
        return {
            'fractal_dimension': float(fractal_dim),
            'entropy_complexity': float(entropy_complexity),
            'fourier_complexity': float(fourier_complexity),
            'overall_complexity_score': float(max(0, min(100, complexity_score)))
        }
    
    def _estimate_fractal_dimension(self, img: np.ndarray) -> float:
        """Estimate fractal dimension using differential box counting"""
        
        try:
            # Convert to binary image
            threshold = np.mean(img)
            binary = (img > threshold).astype(np.uint8)
            
            # Box counting at different scales
            scales = [2, 4, 8, 16, 32]
            counts = []
            
            for scale in scales:
                h, w = binary.shape
                # Downsample image
                if h >= scale and w >= scale:
                    downsampled = binary[::scale, ::scale]
                    # Count non-zero boxes
                    count = np.sum(downsampled > 0)
                    counts.append(count)
                else:
                    break
            
            if len(counts) < 2:
                return 1.5  # Default fractal dimension
            
            # Fit line to log-log plot
            valid_scales = scales[:len(counts)]
            log_scales = np.log([1/s for s in valid_scales])
            log_counts = np.log([c + 1 for c in counts])  # Add 1 to avoid log(0)
            
            if len(log_scales) > 1:
                # Use least squares fit
                A = np.vstack([log_scales, np.ones(len(log_scales))]).T
                slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
                fractal_dimension = abs(slope)
                
                # Clamp to reasonable range
                return max(1.0, min(2.0, fractal_dimension))
            
            return 1.5
            
        except Exception as e:
            logger.warning(f"Fractal dimension estimation failed: {e}")
            return 1.5
    
    def _calculate_entropy_complexity(self, img: np.ndarray) -> float:
        """Calculate entropy-based complexity measure"""
        
        # Calculate histogram
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 255), density=True)
        
        # Remove zero entries
        hist = hist[hist > 0]
        
        # Calculate Shannon entropy
        shannon_entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(256)
        normalized_entropy = shannon_entropy / max_entropy
        
        return max(0, min(1, normalized_entropy))
    
    def _calculate_fourier_complexity(self, img: np.ndarray) -> float:
        """Calculate complexity using Fourier transform"""
        
        # Apply FFT
        fft = np.fft.fft2(img)
        magnitude_spectrum = np.abs(fft)
        
        # Shift zero frequency to center
        magnitude_spectrum = np.fft.fftshift(magnitude_spectrum)
        
        # Calculate energy distribution
        total_energy = np.sum(magnitude_spectrum**2)
        
        # Calculate energy in high-frequency components
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Define high-frequency region (outer ring)
        radius_inner = min(h, w) // 8
        radius_outer = min(h, w) // 2
        
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        high_freq_mask = (distances >= radius_inner) & (distances <= radius_outer)
        
        high_freq_energy = np.sum((magnitude_spectrum * high_freq_mask)**2)
        
        # Complexity is ratio of high-frequency to total energy
        complexity = high_freq_energy / total_energy if total_energy > 0 else 0
        
        return max(0, min(1, complexity))
    
    def _analyze_shape_breakup(self, camo_gray: np.ndarray, camo_color: np.ndarray,
                              options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well patterns break up recognizable shapes"""
        
        # Find contours
        edges = cv2.Canny(camo_gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'shape_breakup_score': 0.0, 'contour_count': 0}
        
        # Analyze contour properties
        contour_complexities = []
        contour_sizes = []
        
        for contour in contours:
            if len(contour) < 5:  # Skip very small contours
                continue
            
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 0 and perimeter > 0:
                # Complexity measure (deviation from circle)
                circularity = 4 * np.pi * area / (perimeter**2)
                complexity = 1 - circularity  # Higher complexity for non-circular shapes
                
                contour_complexities.append(complexity)
                contour_sizes.append(area)
        
        if not contour_complexities:
            return {'shape_breakup_score': 0.0, 'contour_count': 0}
        
        # Shape breakup metrics
        avg_complexity = np.mean(contour_complexities)
        size_variance = np.var(contour_sizes) / (np.mean(contour_sizes)**2) if np.mean(contour_sizes) > 0 else 0
        
        # More complex shapes with varied sizes = better breakup
        shape_breakup_score = (avg_complexity * 0.7 + min(size_variance, 1.0) * 0.3) * 100
        
        return {
            'shape_breakup_score': float(max(0, min(100, shape_breakup_score))),
            'contour_count': len(contour_complexities),
            'average_complexity': float(avg_complexity),
            'size_variance': float(size_variance)
        }
    
    def _analyze_spatial_frequencies(self, camo_gray: np.ndarray, 
                                    options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial frequency content"""
        
        # Apply FFT
        fft = np.fft.fft2(camo_gray)
        magnitude_spectrum = np.abs(fft)
        
        # Shift zero frequency to center
        magnitude_spectrum = np.fft.fftshift(magnitude_spectrum)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Create frequency bands
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        max_distance = min(h, w) // 2
        
        # Define frequency bands
        bands = {
            'low': distances <= max_distance * 0.2,
            'medium': (distances > max_distance * 0.2) & (distances <= max_distance * 0.6),
            'high': distances > max_distance * 0.6
        }
        
        # Calculate energy in each band
        total_energy = np.sum(magnitude_spectrum**2)
        band_energies = {}
        
        for band_name, mask in bands.items():
            band_energy = np.sum((magnitude_spectrum * mask)**2)
            band_energies[band_name] = float(band_energy / total_energy) if total_energy > 0 else 0
        
        # Frequency diversity (more uniform distribution = better disruption)
        energy_values = list(band_energies.values())
        frequency_diversity = 1 - np.var(energy_values)  # Lower variance = more uniform
        
        return {
            'band_energies': band_energies,
            'frequency_diversity': float(max(0, frequency_diversity)),
            'dominant_frequency_band': max(band_energies, key=band_energies.get)
        }
    
    def _calculate_pattern_disruption_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall pattern disruption score"""
        
        scores = []
        weights = []
        
        # Feature analysis contribution
        feature_analysis = analysis_results.get('feature_analysis', {})
        feature_scores = []
        for detector_name, detector_results in feature_analysis.items():
            if 'matching_analysis' in detector_results:
                disruption_score = detector_results['matching_analysis'].get('disruption_score', 0)
                feature_scores.append(disruption_score)
        
        if feature_scores:
            scores.append(np.mean(feature_scores))
            weights.append(0.25)
        
        # Edge analysis contribution
        edge_analysis = analysis_results.get('edge_analysis', {})
        if 'overall' in edge_analysis:
            edge_score = edge_analysis['overall'].get('edge_disruption_score', 0)
            scores.append(edge_score)
            weights.append(0.20)
        
        # Texture analysis contribution
        texture_analysis = analysis_results.get('texture_analysis', {})
        if 'overall' in texture_analysis:
            texture_score = texture_analysis['overall'].get('texture_similarity_score', 0)
            scores.append(texture_score)
            weights.append(0.20)
        
        # Pattern complexity contribution
        complexity_analysis = analysis_results.get('complexity_analysis', {})
        complexity_score = complexity_analysis.get('overall_complexity_score', 0)
        scores.append(complexity_score)
        weights.append(0.15)
        
        # Shape breakup contribution
        shape_analysis = analysis_results.get('shape_analysis', {})
        shape_score = shape_analysis.get('shape_breakup_score', 0)
        scores.append(shape_score)
        weights.append(0.15)
        
        # Frequency analysis contribution
        frequency_analysis = analysis_results.get('frequency_analysis', {})
        freq_diversity = frequency_analysis.get('frequency_diversity', 0)
        scores.append(freq_diversity * 100)
        weights.append(0.05)
        
        # Calculate weighted average
        if scores and weights:
            # Normalize weights
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
            'pipeline_name': 'pattern_disruption',
            'error': error_message,
            'score': 0.0,
            'execution_time': 0.0
        }