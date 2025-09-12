from typing import Dict, Any, List, Optional, Union
import numpy as np
import time
from datetime import datetime
import logging

from .router import HideSeekAnalysisRouter
from ..core.data_manager import TestDataManager
from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('pipeline_controller')


class PipelineController:
    """
    Controls execution of analysis pipelines and manages the complete analysis workflow.
    Coordinates between the router, individual analyzers, and data management.
    """
    
    def __init__(self, data_manager: TestDataManager = None):
        self.router = HideSeekAnalysisRouter()
        self.data_manager = data_manager or TestDataManager()
        
        # Analysis execution options
        self.default_options = {
            'save_intermediate': True,
            'enable_caching': True,
            'parallel_execution': False,  # Future enhancement
            'quality_mode': 'standard'  # 'fast', 'standard', 'detailed'
        }
        
        logger.info("PipelineController initialized")
    
    def execute_full_analysis(self, img: np.ndarray, reference_img: np.ndarray = None, 
                             options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute complete camouflage analysis with all appropriate pipelines.
        
        Args:
            img: Camouflage image to analyze
            reference_img: Optional reference/background image
            options: Analysis options
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        logger.info("Starting full camouflage analysis")
        
        # Merge options with defaults
        analysis_options = {**self.default_options, **(options or {})}
        
        try:
            # Step 1: Analyze image characteristics
            logger.debug("Step 1: Analyzing image characteristics")
            img_characteristics = self.router.analyze_image_characteristics(img)
            
            if analysis_options['save_intermediate']:
                self.data_manager.save_intermediate_results(
                    img_characteristics, 'image_characteristics'
                )
            
            # Step 2: Determine environment type
            logger.debug("Step 2: Determining environment type")
            environment_type = self.router.determine_environment_type(img)
            
            # Step 3: Select appropriate analysis pipelines
            logger.debug("Step 3: Selecting analysis pipelines")
            selected_pipelines = self.router.select_analysis_pipelines(img_characteristics)
            
            # Add environment-specific pipelines if reference image provided
            if reference_img is not None:
                if 'environmental_context' not in selected_pipelines:
                    selected_pipelines.append('environmental_context')
            
            # Step 4: Execute selected pipelines
            logger.info(f"Step 4: Executing {len(selected_pipelines)} analysis pipelines")
            pipeline_results = []
            
            for pipeline_name in selected_pipelines:
                logger.debug(f"Executing pipeline: {pipeline_name}")
                
                try:
                    # Prepare pipeline-specific options
                    pipeline_options = self._prepare_pipeline_options(
                        pipeline_name, analysis_options, img_characteristics, environment_type
                    )
                    
                    # Execute pipeline
                    result = self.router.route_to_pipeline(
                        img, pipeline_name, reference_img, pipeline_options
                    )
                    
                    # Add metadata
                    result['pipeline_name'] = pipeline_name
                    result['execution_time'] = time.time() - start_time
                    
                    pipeline_results.append(result)
                    
                    # Save intermediate results
                    if analysis_options['save_intermediate']:
                        self.data_manager.save_intermediate_results(
                            result, f'pipeline_{pipeline_name}'
                        )
                    
                    logger.debug(f"Pipeline {pipeline_name} completed: score={result.get('score', 0):.1f}")
                    
                except Exception as e:
                    logger.error(f"Pipeline {pipeline_name} failed: {str(e)}")
                    error_result = {
                        'pipeline_name': pipeline_name,
                        'error': str(e),
                        'score': 0.0
                    }
                    pipeline_results.append(error_result)
            
            # Step 5: Aggregate results
            logger.debug("Step 5: Aggregating pipeline results")
            final_results = self.router.aggregate_pipeline_results(
                pipeline_results, environment_type
            )
            
            # Add execution metadata
            execution_time = time.time() - start_time
            final_results['execution_metadata'] = {
                'total_execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'analysis_options': analysis_options,
                'image_characteristics': img_characteristics,
                'selected_pipelines': selected_pipelines
            }
            
            logger.info(f"Full analysis completed in {execution_time:.2f}s")
            logger.info(f"Overall effectiveness score: {final_results['overall_score']:.1f}/100")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Full analysis failed: {str(e)}")
            return {
                'error': str(e),
                'overall_score': 0.0,
                'execution_metadata': {
                    'total_execution_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'failed': True
                }
            }
    
    def execute_quick_analysis(self, img: np.ndarray, 
                              options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute quick analysis with essential pipelines only.
        
        Args:
            img: Camouflage image to analyze
            options: Analysis options
            
        Returns:
            Quick analysis results
        """
        start_time = time.time()
        logger.info("Starting quick camouflage analysis")
        
        # Override options for quick analysis
        quick_options = {
            'save_intermediate': False,
            'enable_caching': True,
            'quality_mode': 'fast'
        }
        quick_options.update(options or {})
        
        try:
            # Essential analysis only
            img_characteristics = self.router.analyze_image_characteristics(img)
            environment_type = self.router.determine_environment_type(img)
            
            # Execute core pipelines only
            core_pipelines = ['color_blending', 'brightness_matching']
            pipeline_results = []
            
            for pipeline_name in core_pipelines:
                try:
                    pipeline_options = self._prepare_pipeline_options(
                        pipeline_name, quick_options, img_characteristics, environment_type
                    )
                    
                    result = self.router.route_to_pipeline(
                        img, pipeline_name, None, pipeline_options
                    )
                    result['pipeline_name'] = pipeline_name
                    pipeline_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Quick pipeline {pipeline_name} failed: {str(e)}")
                    pipeline_results.append({
                        'pipeline_name': pipeline_name,
                        'error': str(e),
                        'score': 0.0
                    })
            
            # Aggregate results
            final_results = self.router.aggregate_pipeline_results(
                pipeline_results, environment_type
            )
            
            execution_time = time.time() - start_time
            final_results['execution_metadata'] = {
                'total_execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'analysis_mode': 'quick',
                'pipelines_executed': len(pipeline_results)
            }
            
            logger.info(f"Quick analysis completed in {execution_time:.2f}s")
            logger.info(f"Quick score estimate: {final_results['overall_score']:.1f}/100")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Quick analysis failed: {str(e)}")
            return {
                'error': str(e),
                'overall_score': 0.0,
                'execution_metadata': {
                    'total_execution_time': time.time() - start_time,
                    'analysis_mode': 'quick',
                    'failed': True
                }
            }
    
    def execute_detailed_analysis(self, img: np.ndarray, reference_img: np.ndarray = None,
                                 options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute detailed analysis with all available pipelines and enhanced options.
        
        Args:
            img: Camouflage image to analyze
            reference_img: Optional reference/background image
            options: Analysis options
            
        Returns:
            Detailed analysis results
        """
        start_time = time.time()
        logger.info("Starting detailed camouflage analysis")
        
        # Enhanced options for detailed analysis
        detailed_options = {
            'save_intermediate': True,
            'enable_caching': True,
            'quality_mode': 'detailed',
            'include_debug_info': True,
            'generate_visualizations': True,
            'statistical_analysis': True
        }
        detailed_options.update(options or {})
        
        try:
            # Enhanced image analysis
            img_characteristics = self.router.analyze_image_characteristics(img)
            environment_type = self.router.determine_environment_type(img)
            
            # Force all pipelines for detailed analysis
            all_pipelines = [
                'color_blending', 'pattern_disruption', 'brightness_matching', 
                'distance_detection', 'environmental_context'
            ]
            
            pipeline_results = []
            pipeline_timings = {}
            
            for pipeline_name in all_pipelines:
                pipeline_start = time.time()
                logger.debug(f"Executing detailed pipeline: {pipeline_name}")
                
                try:
                    pipeline_options = self._prepare_pipeline_options(
                        pipeline_name, detailed_options, img_characteristics, environment_type
                    )
                    
                    result = self.router.route_to_pipeline(
                        img, pipeline_name, reference_img, pipeline_options
                    )
                    
                    pipeline_time = time.time() - pipeline_start
                    pipeline_timings[pipeline_name] = pipeline_time
                    
                    result['pipeline_name'] = pipeline_name
                    result['execution_time'] = pipeline_time
                    
                    # Add detailed metadata for this pipeline
                    if detailed_options.get('include_debug_info'):
                        result['debug_info'] = {
                            'options_used': pipeline_options,
                            'execution_time': pipeline_time
                        }
                    
                    pipeline_results.append(result)
                    
                    # Save detailed intermediate results
                    self.data_manager.save_intermediate_results(
                        result, f'detailed_{pipeline_name}'
                    )
                    
                    logger.debug(f"Detailed pipeline {pipeline_name} completed in {pipeline_time:.2f}s")
                    
                except Exception as e:
                    pipeline_time = time.time() - pipeline_start
                    pipeline_timings[pipeline_name] = pipeline_time
                    
                    logger.error(f"Detailed pipeline {pipeline_name} failed: {str(e)}")
                    pipeline_results.append({
                        'pipeline_name': pipeline_name,
                        'error': str(e),
                        'score': 0.0,
                        'execution_time': pipeline_time
                    })
            
            # Enhanced aggregation with statistical analysis
            final_results = self.router.aggregate_pipeline_results(
                pipeline_results, environment_type
            )
            
            # Add detailed statistics
            if detailed_options.get('statistical_analysis'):
                final_results['statistical_analysis'] = self._generate_statistical_analysis(
                    pipeline_results, img_characteristics
                )
            
            # Add comprehensive execution metadata
            execution_time = time.time() - start_time
            final_results['execution_metadata'] = {
                'total_execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'analysis_mode': 'detailed',
                'pipeline_timings': pipeline_timings,
                'image_characteristics': img_characteristics,
                'environment_type': environment_type,
                'options_used': detailed_options
            }
            
            # Generate enhanced recommendations
            final_results['enhanced_recommendations'] = self._generate_enhanced_recommendations(
                final_results, img_characteristics
            )
            
            logger.info(f"Detailed analysis completed in {execution_time:.2f}s")
            logger.info(f"Detailed effectiveness score: {final_results['overall_score']:.1f}/100")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Detailed analysis failed: {str(e)}")
            return {
                'error': str(e),
                'overall_score': 0.0,
                'execution_metadata': {
                    'total_execution_time': time.time() - start_time,
                    'analysis_mode': 'detailed',
                    'failed': True
                }
            }
    
    def execute_comparison_analysis(self, images: List[np.ndarray], 
                                   reference_img: np.ndarray = None,
                                   labels: List[str] = None,
                                   options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute comparative analysis of multiple camouflage patterns.
        
        Args:
            images: List of camouflage images to compare
            reference_img: Optional common reference image
            labels: Optional labels for each image
            options: Analysis options
            
        Returns:
            Comparative analysis results
        """
        start_time = time.time()
        logger.info(f"Starting comparison analysis of {len(images)} images")
        
        if labels is None:
            labels = [f"Pattern_{i+1}" for i in range(len(images))]
        
        comparison_options = {
            'save_intermediate': True,
            'quality_mode': 'standard'
        }
        comparison_options.update(options or {})
        
        try:
            individual_results = []
            
            # Analyze each image
            for idx, (img, label) in enumerate(zip(images, labels)):
                logger.debug(f"Analyzing image {idx+1}/{len(images)}: {label}")
                
                result = self.execute_full_analysis(img, reference_img, comparison_options)
                result['label'] = label
                result['index'] = idx
                individual_results.append(result)
            
            # Generate comparison analysis
            comparison_results = {
                'individual_results': individual_results,
                'comparison_summary': self._generate_comparison_summary(individual_results),
                'rankings': self._generate_rankings(individual_results),
                'statistical_comparison': self._generate_statistical_comparison(individual_results),
                'execution_metadata': {
                    'total_execution_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_mode': 'comparison',
                    'images_analyzed': len(images),
                    'labels': labels
                }
            }
            
            logger.info(f"Comparison analysis completed in {time.time() - start_time:.2f}s")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Comparison analysis failed: {str(e)}")
            return {
                'error': str(e),
                'execution_metadata': {
                    'total_execution_time': time.time() - start_time,
                    'analysis_mode': 'comparison',
                    'failed': True
                }
            }
    
    def _prepare_pipeline_options(self, pipeline_name: str, analysis_options: Dict[str, Any],
                                 img_characteristics: Dict[str, Any], 
                                 environment_type: str) -> Dict[str, Any]:
        """Prepare pipeline-specific options based on analysis context"""
        base_options = {
            'quality_mode': analysis_options.get('quality_mode', 'standard'),
            'environment_type': environment_type,
            'image_characteristics': img_characteristics
        }
        
        # Pipeline-specific option adjustments
        if pipeline_name == 'color_blending':
            base_options.update({
                'color_space': 'LAB',
                'use_perceptual_metrics': True,
                'delta_e_method': 'CIE2000'
            })
        
        elif pipeline_name == 'pattern_disruption':
            texture_complexity = img_characteristics.get('texture_properties', {}).get('texture_energy', 0.5)
            base_options.update({
                'feature_detector': 'ORB' if texture_complexity < 0.5 else 'SIFT',
                'analyze_fractal_dimension': analysis_options.get('quality_mode') == 'detailed'
            })
        
        elif pipeline_name == 'brightness_matching':
            base_options.update({
                'analyze_local_contrast': True,
                'test_illumination_conditions': analysis_options.get('quality_mode') != 'fast'
            })
        
        elif pipeline_name == 'distance_detection':
            resolution = img_characteristics.get('dimensions', (640, 480))
            base_options.update({
                'test_distances': config.get_standard_distances(),
                'simulate_atmospheric_effects': min(resolution) > 400
            })
        
        elif pipeline_name == 'environmental_context':
            base_options.update({
                'test_multiple_environments': analysis_options.get('quality_mode') == 'detailed',
                'seasonal_analysis': analysis_options.get('quality_mode') == 'detailed'
            })
        
        return base_options
    
    def _generate_statistical_analysis(self, pipeline_results: List[Dict[str, Any]], 
                                     img_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis of pipeline results"""
        scores = [result.get('score', 0) for result in pipeline_results if 'error' not in result]
        
        if not scores:
            return {'error': 'No valid scores for statistical analysis'}
        
        return {
            'score_statistics': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'range': np.max(scores) - np.min(scores),
                'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
            },
            'consistency_metrics': {
                'score_consistency': 1.0 - (np.std(scores) / 100.0),  # Higher is more consistent
                'performance_balance': 1.0 - ((np.max(scores) - np.min(scores)) / 100.0)
            },
            'pipeline_performance': {
                'successful_pipelines': len([r for r in pipeline_results if 'error' not in r]),
                'failed_pipelines': len([r for r in pipeline_results if 'error' in r]),
                'average_execution_time': np.mean([r.get('execution_time', 0) for r in pipeline_results])
            }
        }
    
    def _generate_enhanced_recommendations(self, results: Dict[str, Any], 
                                         img_characteristics: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on detailed analysis"""
        recommendations = results.get('recommendations', [])
        
        # Add detailed analysis specific recommendations
        component_scores = results.get('component_scores', {})
        
        # Environment-specific recommendations
        environment_type = results.get('analysis_metadata', {}).get('environment_type')
        if environment_type:
            env_config = config.get_environment_config(environment_type)
            if env_config:
                recommendations.append(
                    f"For {environment_type} environments, focus on {env_config.get('primary_focus', 'overall balance')}"
                )
        
        # Image characteristic-based recommendations
        complexity = img_characteristics.get('complexity_score', 0.5)
        if complexity < 0.3:
            recommendations.append("Consider adding more pattern complexity for better disruption")
        elif complexity > 0.8:
            recommendations.append("Excellent pattern complexity - maintain this level across different scales")
        
        # Statistical analysis recommendations
        statistical_analysis = results.get('statistical_analysis', {})
        consistency = statistical_analysis.get('consistency_metrics', {}).get('score_consistency', 1.0)
        if consistency < 0.7:
            recommendations.append("Improve consistency across different analysis components")
        
        return recommendations
    
    def _generate_comparison_summary(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of comparison analysis"""
        valid_results = [r for r in individual_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results for comparison'}
        
        overall_scores = [r['overall_score'] for r in valid_results]
        
        return {
            'best_performer': {
                'label': valid_results[np.argmax(overall_scores)]['label'],
                'score': np.max(overall_scores)
            },
            'worst_performer': {
                'label': valid_results[np.argmin(overall_scores)]['label'], 
                'score': np.min(overall_scores)
            },
            'average_score': np.mean(overall_scores),
            'score_range': np.max(overall_scores) - np.min(overall_scores),
            'performance_consistency': 1.0 - (np.std(overall_scores) / 100.0)
        }
    
    def _generate_rankings(self, individual_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate rankings for different metrics"""
        valid_results = [r for r in individual_results if 'error' not in r]
        
        rankings = {}
        
        # Overall score ranking
        overall_ranking = sorted(valid_results, key=lambda x: x['overall_score'], reverse=True)
        rankings['overall'] = [
            {'rank': i+1, 'label': r['label'], 'score': r['overall_score']}
            for i, r in enumerate(overall_ranking)
        ]
        
        # Component rankings
        components = ['color', 'pattern', 'brightness', 'distance']
        for component in components:
            component_results = [r for r in valid_results if component in r.get('component_scores', {})]
            if component_results:
                component_ranking = sorted(
                    component_results, 
                    key=lambda x: x['component_scores'][component], 
                    reverse=True
                )
                rankings[component] = [
                    {'rank': i+1, 'label': r['label'], 'score': r['component_scores'][component]}
                    for i, r in enumerate(component_ranking)
                ]
        
        return rankings
    
    def _generate_statistical_comparison(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistical comparison of multiple results"""
        valid_results = [r for r in individual_results if 'error' not in r]
        
        if len(valid_results) < 2:
            return {'error': 'Need at least 2 valid results for statistical comparison'}
        
        # Extract component scores for all results
        components = ['color', 'pattern', 'brightness', 'distance']
        component_stats = {}
        
        for component in components:
            scores = []
            for result in valid_results:
                if component in result.get('component_scores', {}):
                    scores.append(result['component_scores'][component])
            
            if scores:
                component_stats[component] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
                }
        
        return {
            'component_statistics': component_stats,
            'sample_size': len(valid_results),
            'most_consistent_component': min(component_stats.items(), 
                                           key=lambda x: x[1]['coefficient_of_variation'])[0] if component_stats else None,
            'most_variable_component': max(component_stats.items(), 
                                         key=lambda x: x[1]['coefficient_of_variation'])[0] if component_stats else None
        }