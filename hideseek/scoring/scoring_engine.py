import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('scoring_engine')


class HideSeekScoringEngine:
    """
    Advanced scoring engine for comprehensive camouflage effectiveness evaluation.
    Provides weighted scoring, statistical analysis, and visualization capabilities.
    """
    
    def __init__(self):
        # Load scoring configuration
        self.default_weights = config.get_scoring_weights()
        self.environment_adjustments = config.get('scoring.environment_adjustments', {})
        
        # Statistical parameters
        self.confidence_level = config.get('output.confidence_level', 0.95)
        self.decimal_precision = config.get('output.decimal_precision', 2)
        
        # Visualization settings
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        logger.info("ScoringEngine initialized with adaptive weighting system")
    
    def calculate_weighted_score(self, scores: Dict[str, float], environment: str = None) -> Dict[str, Any]:
        """
        Calculate overall effectiveness score with adaptive weights.
        
        Args:
            scores: Dictionary of component scores
            environment: Environment type for weight adjustment
            
        Returns:
            Comprehensive scoring results
        """
        logger.debug(f"Calculating weighted score for environment: {environment}")
        
        # Get base weights
        weights = self.default_weights.copy()
        
        # Apply environment-specific adjustments
        if environment and environment in self.environment_adjustments:
            env_adjustments = self.environment_adjustments[environment]
            for component, adjustment in env_adjustments.items():
                if component in weights:
                    weights[component] = adjustment
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        weighted_components = {}
        total_score = 0.0
        
        # Map score keys to weight keys
        score_weight_mapping = {
            'color': 'color_blending',
            'color_blending': 'color_blending',
            'pattern': 'pattern_disruption',
            'pattern_disruption': 'pattern_disruption',
            'brightness': 'brightness_matching',
            'brightness_matching': 'brightness_matching',
            'distance': 'distance_effectiveness',
            'distance_detection': 'distance_effectiveness',
            'environmental_context': 'environmental_adaptability'
        }
        
        for score_key, score_value in scores.items():
            weight_key = score_weight_mapping.get(score_key, score_key)
            weight = weights.get(weight_key, 0.0)
            
            if weight > 0:
                weighted_score = score_value * weight
                weighted_components[score_key] = {
                    'raw_score': round(float(score_value), self.decimal_precision),
                    'weight': round(float(weight), 3),
                    'weighted_score': round(float(weighted_score), self.decimal_precision)
                }
                total_score += weighted_score
        
        # Calculate score statistics
        raw_scores = [s for s in scores.values() if s is not None]
        score_stats = self._calculate_score_statistics(raw_scores) if raw_scores else {}
        
        # Determine overall rating
        overall_rating = self._determine_rating(total_score)
        
        result = {
            'overall_score': round(float(total_score), self.decimal_precision),
            'overall_rating': overall_rating,
            'weighted_components': weighted_components,
            'weights_used': weights,
            'environment': environment,
            'score_statistics': score_stats,
            'calculation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'components_analyzed': len(weighted_components),
                'total_weight': round(sum(weights.values()), 3)
            }
        }
        
        logger.info(f"Weighted score calculated: {total_score:.1f}/100 ({overall_rating})")
        return result
    
    def _calculate_score_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for scores"""
        
        scores_array = np.array(scores)
        
        return {
            'mean': round(float(np.mean(scores_array)), self.decimal_precision),
            'median': round(float(np.median(scores_array)), self.decimal_precision),
            'std': round(float(np.std(scores_array)), self.decimal_precision),
            'min': round(float(np.min(scores_array)), self.decimal_precision),
            'max': round(float(np.max(scores_array)), self.decimal_precision),
            'range': round(float(np.max(scores_array) - np.min(scores_array)), self.decimal_precision),
            'coefficient_of_variation': round(float(np.std(scores_array) / (np.mean(scores_array) + 1e-6)), 3)
        }
    
    def _determine_rating(self, score: float) -> str:
        """Determine qualitative rating from numerical score"""
        
        if score >= 90:
            return "Exceptional"
        elif score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Very Good"
        elif score >= 60:
            return "Good"
        elif score >= 50:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Very Poor"
    
    def generate_detailed_breakdown(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed score breakdown with component analysis.
        
        Args:
            all_metrics: Complete analysis results
            
        Returns:
            Detailed breakdown analysis
        """
        logger.debug("Generating detailed score breakdown")
        
        breakdown = {
            'component_analysis': {},
            'performance_insights': {},
            'improvement_recommendations': [],
            'statistical_summary': {}
        }
        
        # Extract component scores
        component_scores = all_metrics.get('component_scores', {})
        
        # Analyze each component
        for component, score in component_scores.items():
            component_analysis = self._analyze_component_performance(
                component, score, all_metrics
            )
            breakdown['component_analysis'][component] = component_analysis
        
        # Generate performance insights
        breakdown['performance_insights'] = self._generate_performance_insights(component_scores)
        
        # Generate improvement recommendations
        breakdown['improvement_recommendations'] = self._generate_improvement_recommendations(
            component_scores, all_metrics
        )
        
        # Statistical summary
        if component_scores:
            scores_list = list(component_scores.values())
            breakdown['statistical_summary'] = self._calculate_score_statistics(scores_list)
            
            # Add distribution analysis
            breakdown['score_distribution'] = self._analyze_score_distribution(scores_list)
        
        return breakdown
    
    def _analyze_component_performance(self, component: str, score: float, 
                                      all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual component performance"""
        
        analysis = {
            'score': round(float(score), self.decimal_precision),
            'rating': self._determine_rating(score),
            'percentile': self._calculate_percentile_rank(score),
            'detailed_metrics': {},
            'strengths': [],
            'weaknesses': []
        }
        
        # Extract detailed metrics for this component
        detailed_results = all_metrics.get('detailed_results', {})
        
        if component == 'color' or component == 'color_blending':
            color_results = detailed_results.get('color_blending', {})
            analysis['detailed_metrics'] = self._extract_color_metrics(color_results)
            analysis['strengths'], analysis['weaknesses'] = self._analyze_color_performance(color_results)
            
        elif component == 'pattern' or component == 'pattern_disruption':
            pattern_results = detailed_results.get('pattern_disruption', {})
            analysis['detailed_metrics'] = self._extract_pattern_metrics(pattern_results)
            analysis['strengths'], analysis['weaknesses'] = self._analyze_pattern_performance(pattern_results)
            
        elif component == 'brightness' or component == 'brightness_matching':
            brightness_results = detailed_results.get('brightness_matching', {})
            analysis['detailed_metrics'] = self._extract_brightness_metrics(brightness_results)
            analysis['strengths'], analysis['weaknesses'] = self._analyze_brightness_performance(brightness_results)
            
        elif component == 'distance' or component == 'distance_detection':
            distance_results = detailed_results.get('distance_detection', {})
            analysis['detailed_metrics'] = self._extract_distance_metrics(distance_results)
            analysis['strengths'], analysis['weaknesses'] = self._analyze_distance_performance(distance_results)
            
        elif component == 'environmental_context':
            env_results = detailed_results.get('environmental_context', {})
            analysis['detailed_metrics'] = self._extract_environmental_metrics(env_results)
            analysis['strengths'], analysis['weaknesses'] = self._analyze_environmental_performance(env_results)
        
        return analysis
    
    def _calculate_percentile_rank(self, score: float) -> int:
        """Calculate percentile rank for score (0-100 scale)"""
        # Simple percentile calculation
        return min(100, max(0, int(score)))
    
    def _extract_color_metrics(self, color_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from color analysis"""
        color_analysis = color_results.get('color_analysis', {})
        return {
            'delta_e_mean': color_analysis.get('delta_e_statistics', {}).get('mean_delta_e', 0),
            'delta_e_min': color_analysis.get('delta_e_statistics', {}).get('min_delta_e', 0),
            'color_matching_score': color_analysis.get('color_matching_score', 0)
        }
    
    def _analyze_color_performance(self, color_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze color performance strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        color_analysis = color_results.get('color_analysis', {})
        delta_e_stats = color_analysis.get('delta_e_statistics', {})
        
        if delta_e_stats.get('mean_delta_e', 100) < 5.0:
            strengths.append("Excellent color matching (Delta-E < 5)")
        elif delta_e_stats.get('mean_delta_e', 100) > 15.0:
            weaknesses.append("Poor color matching (Delta-E > 15)")
        
        quality_dist = color_analysis.get('quality_distribution', {})
        excellent_percent = quality_dist.get('excellent', 0)
        
        if excellent_percent > 70:
            strengths.append("High percentage of excellent color matches")
        elif excellent_percent < 30:
            weaknesses.append("Low percentage of excellent color matches")
        
        return strengths, weaknesses
    
    def _extract_pattern_metrics(self, pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from pattern analysis"""
        feature_analysis = pattern_results.get('feature_analysis', {})
        complexity_analysis = pattern_results.get('complexity_analysis', {})
        
        return {
            'fractal_dimension': complexity_analysis.get('fractal_dimension', 0),
            'overall_complexity': complexity_analysis.get('overall_complexity_score', 0),
            'feature_disruption': self._calculate_average_feature_disruption(feature_analysis)
        }
    
    def _calculate_average_feature_disruption(self, feature_analysis: Dict[str, Any]) -> float:
        """Calculate average feature disruption score"""
        disruption_scores = []
        
        for detector, results in feature_analysis.items():
            if isinstance(results, dict) and 'matching_analysis' in results:
                score = results['matching_analysis'].get('disruption_score', 0)
                disruption_scores.append(score)
        
        return np.mean(disruption_scores) if disruption_scores else 0.0
    
    def _analyze_pattern_performance(self, pattern_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze pattern performance strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        complexity_analysis = pattern_results.get('complexity_analysis', {})
        complexity_score = complexity_analysis.get('overall_complexity_score', 0)
        
        if complexity_score > 75:
            strengths.append("High pattern complexity for effective disruption")
        elif complexity_score < 40:
            weaknesses.append("Low pattern complexity may be insufficient")
        
        fractal_dim = complexity_analysis.get('fractal_dimension', 1.5)
        if 1.3 <= fractal_dim <= 1.7:
            strengths.append("Natural fractal dimension for realistic patterns")
        
        return strengths, weaknesses
    
    def _extract_brightness_metrics(self, brightness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from brightness analysis"""
        brightness_analysis = brightness_results.get('brightness_analysis', {})
        illumination_analysis = brightness_results.get('illumination_analysis', {})
        
        return {
            'brightness_matching': brightness_analysis.get('brightness_matching_score', 0),
            'illumination_adaptability': illumination_analysis.get('overall_adaptability_score', 0),
            'shadow_effectiveness': brightness_results.get('shadow_analysis', {}).get('shadow_effectiveness_score', 0)
        }
    
    def _analyze_brightness_performance(self, brightness_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze brightness performance strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        shadow_analysis = brightness_results.get('shadow_analysis', {})
        shadow_score = shadow_analysis.get('shadow_effectiveness_score', 0)
        
        if shadow_score > 75:
            strengths.append("Excellent shadow pattern realism")
        elif shadow_score < 40:
            weaknesses.append("Poor shadow pattern effectiveness")
        
        illumination_analysis = brightness_results.get('illumination_analysis', {})
        adaptability = illumination_analysis.get('overall_adaptability_score', 0)
        
        if adaptability > 70:
            strengths.append("Good adaptation across lighting conditions")
        elif adaptability < 40:
            weaknesses.append("Poor performance in varied lighting")
        
        return strengths, weaknesses
    
    def _extract_distance_metrics(self, distance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from distance analysis"""
        return {
            'critical_distance': distance_results.get('critical_distance', 0),
            'effectiveness_range': distance_results.get('effectiveness_analysis', {}).get('effective_range', 0),
            'long_range_performance': distance_results.get('effectiveness_analysis', {}).get('range_effectiveness', 0)
        }
    
    def _analyze_distance_performance(self, distance_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze distance performance strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        critical_distance = distance_results.get('critical_distance', 0)
        
        if critical_distance > 75:
            strengths.append("Excellent long-range effectiveness")
        elif critical_distance < 25:
            weaknesses.append("Poor long-range performance")
        
        effectiveness_analysis = distance_results.get('effectiveness_analysis', {})
        range_effectiveness = effectiveness_analysis.get('range_effectiveness', 0)
        
        if range_effectiveness > 60:
            strengths.append("Good effectiveness across distance range")
        elif range_effectiveness < 30:
            weaknesses.append("Limited effective range")
        
        return strengths, weaknesses
    
    def _extract_environmental_metrics(self, env_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from environmental analysis"""
        environment_compatibility = env_results.get('environment_compatibility', {})
        versatility_matrix = env_results.get('versatility_matrix', {})
        
        return {
            'average_compatibility': environment_compatibility.get('average_compatibility', 0),
            'versatility_score': versatility_matrix.get('versatility_metrics', {}).get('average_score', 0),
            'best_environment': environment_compatibility.get('best_environment', 'unknown')
        }
    
    def _analyze_environmental_performance(self, env_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze environmental performance strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        versatility_matrix = env_results.get('versatility_matrix', {})
        if 'versatility_rating' in versatility_matrix:
            rating = versatility_matrix['versatility_rating']
            if rating == 'highly_versatile':
                strengths.append("Highly versatile across multiple environments")
            elif rating == 'environment_specific':
                weaknesses.append("Limited to specific environmental conditions")
        
        return strengths, weaknesses
    
    def _generate_performance_insights(self, component_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate performance insights from component scores"""
        
        if not component_scores:
            return {}
        
        scores = list(component_scores.values())
        best_component = max(component_scores.items(), key=lambda x: x[1])
        worst_component = min(component_scores.items(), key=lambda x: x[1])
        
        return {
            'strongest_aspect': {
                'component': best_component[0],
                'score': round(best_component[1], self.decimal_precision),
                'rating': self._determine_rating(best_component[1])
            },
            'weakest_aspect': {
                'component': worst_component[0],
                'score': round(worst_component[1], self.decimal_precision),
                'rating': self._determine_rating(worst_component[1])
            },
            'score_consistency': {
                'coefficient_of_variation': round(np.std(scores) / (np.mean(scores) + 1e-6), 3),
                'range': round(max(scores) - min(scores), self.decimal_precision),
                'interpretation': self._interpret_consistency(scores)
            },
            'overall_balance': self._assess_overall_balance(component_scores)
        }
    
    def _interpret_consistency(self, scores: List[float]) -> str:
        """Interpret score consistency"""
        cv = np.std(scores) / (np.mean(scores) + 1e-6)
        
        if cv < 0.15:
            return "Very consistent performance across components"
        elif cv < 0.25:
            return "Good consistency with minor variations"
        elif cv < 0.4:
            return "Moderate consistency with some significant differences"
        else:
            return "Inconsistent performance with major variations"
    
    def _assess_overall_balance(self, component_scores: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall balance of camouflage performance"""
        
        scores = list(component_scores.values())
        mean_score = np.mean(scores)
        
        # Count components above/below average
        above_avg = sum(1 for s in scores if s > mean_score)
        below_avg = sum(1 for s in scores if s < mean_score)
        
        # Assess balance
        if abs(above_avg - below_avg) <= 1:
            balance_rating = "Well balanced"
        elif abs(above_avg - below_avg) <= 2:
            balance_rating = "Moderately balanced"
        else:
            balance_rating = "Unbalanced"
        
        return {
            'rating': balance_rating,
            'components_above_average': above_avg,
            'components_below_average': below_avg,
            'average_score': round(mean_score, self.decimal_precision)
        }
    
    def _generate_improvement_recommendations(self, component_scores: Dict[str, float], 
                                            all_metrics: Dict[str, Any]) -> List[str]:
        """Generate specific improvement recommendations"""
        
        recommendations = []
        
        # Sort components by score (lowest first for priority)
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1])
        
        # Focus on lowest scoring components
        for component, score in sorted_components[:2]:  # Top 2 lowest
            if score < 60:  # Only for scores below "good" threshold
                recs = self._get_component_recommendations(component, score, all_metrics)
                recommendations.extend(recs)
        
        # Add general recommendations based on overall pattern
        scores = list(component_scores.values())
        if all(s > 80 for s in scores):
            recommendations.append("Excellent overall performance - consider testing in more challenging conditions")
        elif all(s < 50 for s in scores):
            recommendations.append("Consider fundamental redesign focusing on environmental requirements")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_component_recommendations(self, component: str, score: float, 
                                     all_metrics: Dict[str, Any]) -> List[str]:
        """Get specific recommendations for component improvement"""
        
        recommendations = []
        
        if component in ['color', 'color_blending']:
            if score < 50:
                recommendations.append("Improve color matching by adjusting hue and saturation to better match target environment")
            elif score < 70:
                recommendations.append("Fine-tune color blend by reducing Delta-E color differences")
        
        elif component in ['pattern', 'pattern_disruption']:
            if score < 50:
                recommendations.append("Increase pattern complexity and add more disruptive elements")
            elif score < 70:
                recommendations.append("Enhance edge breaking patterns to better disrupt recognizable shapes")
        
        elif component in ['brightness', 'brightness_matching']:
            if score < 50:
                recommendations.append("Adjust brightness levels and contrast to match environmental lighting")
            elif score < 70:
                recommendations.append("Improve shadow patterns for more realistic lighting effects")
        
        elif component in ['distance', 'distance_detection']:
            if score < 50:
                recommendations.append("Enhance pattern for better long-range effectiveness")
            elif score < 70:
                recommendations.append("Improve high-frequency details for better distance performance")
        
        elif component == 'environmental_context':
            if score < 50:
                recommendations.append("Redesign for better environmental versatility or specify target environment")
            elif score < 70:
                recommendations.append("Test and optimize for seasonal variations")
        
        return recommendations
    
    def _analyze_score_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Analyze the distribution of component scores"""
        
        scores_array = np.array(scores)
        
        # Histogram analysis
        hist, bin_edges = np.histogram(scores_array, bins=5, range=(0, 100))
        
        # Determine distribution shape
        skewness = stats.skew(scores_array)
        kurtosis = stats.kurtosis(scores_array)
        
        if abs(skewness) < 0.5:
            distribution_shape = "symmetric"
        elif skewness > 0.5:
            distribution_shape = "right-skewed (most scores below average)"
        else:
            distribution_shape = "left-skewed (most scores above average)"
        
        return {
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'distribution_shape': distribution_shape,
            'skewness': round(float(skewness), 3),
            'kurtosis': round(float(kurtosis), 3)
        }
    
    def create_radar_chart(self, scores: Dict[str, float], title: str = "Camouflage Effectiveness Profile",
                          save_path: str = None) -> plt.Figure:
        """
        Create radar chart visualization of component scores.
        
        Args:
            scores: Dictionary of component scores
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Matplotlib figure object
        """
        logger.debug(f"Creating radar chart with {len(scores)} components")
        
        # Prepare data
        components = list(scores.keys())
        values = list(scores.values())
        
        # Clean component names for display
        display_names = [name.replace('_', ' ').title() for name in components]
        
        # Number of variables
        N = len(components)
        
        # Compute angles for each component
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Close the plot
        values += values[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot the scores
        line = ax.plot(angles, values, 'o-', linewidth=3, label='Camouflage Effectiveness')
        ax.fill(angles, values, alpha=0.25, color=line[0].get_color())
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_names, fontsize=12)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10, alpha=0.7)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add title
        plt.title(title, size=16, fontweight='bold', pad=20)
        
        # Add score annotations
        for angle, value, name in zip(angles[:-1], values[:-1], display_names):
            ax.annotate(f'{value:.1f}', 
                       xy=(angle, value), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=10, 
                       fontweight='bold',
                       ha='center')
        
        # Add legend with overall score
        overall_score = np.mean([v for v in scores.values()])
        legend_text = f'Overall Score: {overall_score:.1f}/100'
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                fontsize=12, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Radar chart saved to {save_path}")
        
        return fig
    
    def generate_comparison_matrix(self, multiple_camos: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate comparison matrix for multiple camouflage patterns.
        
        Args:
            multiple_camos: List of analysis results for different camouflage patterns
            
        Returns:
            DataFrame containing comparison data
        """
        logger.debug(f"Generating comparison matrix for {len(multiple_camos)} patterns")
        
        comparison_data = []
        
        for i, camo_result in enumerate(multiple_camos):
            # Extract basic info
            test_info = camo_result.get('test_info', {})
            name = test_info.get('name', f'Pattern_{i+1}')
            
            # Extract scores
            overall_score = camo_result.get('overall_score', 0)
            component_scores = camo_result.get('component_scores', {})
            
            # Create row data
            row = {
                'Pattern_Name': name,
                'Overall_Score': round(overall_score, self.decimal_precision),
                'Overall_Rating': self._determine_rating(overall_score),
                'Color_Score': round(component_scores.get('color', 0), self.decimal_precision),
                'Pattern_Score': round(component_scores.get('pattern', 0), self.decimal_precision),
                'Brightness_Score': round(component_scores.get('brightness', 0), self.decimal_precision),
                'Distance_Score': round(component_scores.get('distance', 0), self.decimal_precision),
                'Environment_Score': round(component_scores.get('environmental_context', 0), self.decimal_precision),
                'Best_Environment': camo_result.get('analysis_metadata', {}).get('environment_type', 'Unknown'),
                'Rank': 0  # Will be filled after sorting
            }
            
            # Add execution metadata
            execution_meta = camo_result.get('execution_metadata', {})
            row['Analysis_Time'] = round(execution_meta.get('total_execution_time', 0), 1)
            
            comparison_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        if not df.empty:
            # Sort by overall score and assign ranks
            df = df.sort_values('Overall_Score', ascending=False).reset_index(drop=True)
            df['Rank'] = range(1, len(df) + 1)
            
            # Reorder columns
            column_order = ['Rank', 'Pattern_Name', 'Overall_Score', 'Overall_Rating', 
                          'Color_Score', 'Pattern_Score', 'Brightness_Score', 
                          'Distance_Score', 'Environment_Score', 'Best_Environment', 
                          'Analysis_Time']
            df = df[column_order]
        
        logger.info(f"Comparison matrix generated with {len(df)} patterns")
        return df
    
    def export_scientific_report(self, results: Dict[str, Any], filepath: str):
        """
        Generate publication-ready scientific report.
        
        Args:
            results: Complete analysis results
            filepath: Output file path
        """
        logger.info(f"Generating scientific report: {filepath}")
        
        with PdfPages(filepath) as pdf:
            # Title page
            self._create_title_page(results, pdf)
            
            # Executive summary
            self._create_executive_summary(results, pdf)
            
            # Methodology section
            self._create_methodology_section(results, pdf)
            
            # Results section with radar chart
            self._create_results_section(results, pdf)
            
            # Statistical analysis section
            self._create_statistical_section(results, pdf)
            
            # Discussion and recommendations
            self._create_discussion_section(results, pdf)
            
            # Appendices
            self._create_appendices_section(results, pdf)
        
        logger.info(f"Scientific report generated: {filepath}")
    
    def _create_title_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create title page for scientific report"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, 'HideSeek Camouflage Analysis Report', 
               ha='center', va='center', fontsize=24, fontweight='bold')
        
        # Subtitle
        test_info = results.get('test_info', {})
        subtitle = test_info.get('name', 'Camouflage Effectiveness Analysis')
        ax.text(0.5, 0.7, subtitle, ha='center', va='center', fontsize=18)
        
        # Analysis details
        overall_score = results.get('overall_score', 0)
        ax.text(0.5, 0.6, f'Overall Effectiveness: {overall_score:.1f}/100', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Metadata
        ax.text(0.5, 0.4, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
               ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.35, "HideSeek Analysis System v1.0", 
               ha='center', va='center', fontsize=10, style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_executive_summary(self, results: Dict[str, Any], pdf: PdfPages):
        """Create executive summary page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        y_pos = 0.95
        ax.text(0.5, y_pos, 'Executive Summary', ha='center', va='top', 
               fontsize=20, fontweight='bold')
        y_pos -= 0.1
        
        # Key findings
        component_scores = results.get('component_scores', {})
        best_component = max(component_scores.items(), key=lambda x: x[1]) if component_scores else ('N/A', 0)
        worst_component = min(component_scores.items(), key=lambda x: x[1]) if component_scores else ('N/A', 0)
        
        summary_text = f"""
Key Findings:
• Overall camouflage effectiveness: {results.get('overall_score', 0):.1f}/100
• Strongest aspect: {best_component[0].title()} ({best_component[1]:.1f}/100)
• Area for improvement: {worst_component[0].title()} ({worst_component[1]:.1f}/100)
• Primary environment suitability: {results.get('analysis_metadata', {}).get('environment_type', 'Unknown')}

Recommendations:
"""
        
        recommendations = results.get('recommendations', [])
        for i, rec in enumerate(recommendations[:3], 1):
            summary_text += f"• {rec}\n"
        
        ax.text(0.05, y_pos, summary_text, ha='left', va='top', fontsize=12, 
               wrap=True, family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_methodology_section(self, results: Dict[str, Any], pdf: PdfPages):
        """Create methodology section"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        y_pos = 0.95
        ax.text(0.5, y_pos, 'Methodology', ha='center', va='top', 
               fontsize=20, fontweight='bold')
        y_pos -= 0.08
        
        methodology_text = """
Analysis Framework:
The HideSeek system employs a multi-dimensional analysis approach based on established 
camouflage effectiveness principles:

1. Color Blending Analysis
   • Perceptual color difference calculation using CIEDE2000 standard
   • LAB color space conversion for human vision simulation
   • Gamma linearization and white balance correction

2. Pattern Disruption Analysis
   • Multi-feature detection using ORB, SIFT, and BRISK algorithms
   • Texture analysis through Gabor filters and Local Binary Patterns
   • Fractal dimension estimation for pattern complexity assessment

3. Brightness and Contrast Analysis
   • Multi-scale local contrast measurement
   • Shadow pattern detection and realism evaluation
   • Multi-illumination condition testing

4. Distance Effectiveness Simulation
   • Angular size calculations based on human visual acuity
   • Atmospheric scattering and blur simulation
   • Detection probability modeling across distance ranges

5. Environmental Context Analysis
   • Multi-environment compatibility testing
   • Seasonal variation simulation
   • Background complexity matching assessment

Scoring Methodology:
Component scores are weighted and combined using environment-specific adjustments
to produce an overall effectiveness rating from 0-100.
"""
        
        ax.text(0.05, y_pos, methodology_text, ha='left', va='top', fontsize=10, 
               wrap=True, family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_results_section(self, results: Dict[str, Any], pdf: PdfPages):
        """Create results section with radar chart"""
        # Create radar chart
        component_scores = results.get('component_scores', {})
        if component_scores:
            radar_fig = self.create_radar_chart(component_scores, 
                                              "Camouflage Effectiveness Analysis Results")
            pdf.savefig(radar_fig, bbox_inches='tight')
            plt.close()
    
    def _create_statistical_section(self, results: Dict[str, Any], pdf: PdfPages):
        """Create statistical analysis section"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add statistical content here
        ax.text(0.5, 0.9, 'Statistical Analysis', ha='center', va='top', 
               fontsize=20, fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_discussion_section(self, results: Dict[str, Any], pdf: PdfPages):
        """Create discussion section"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.9, 'Discussion & Recommendations', ha='center', va='top', 
               fontsize=20, fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_appendices_section(self, results: Dict[str, Any], pdf: PdfPages):
        """Create appendices section"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.9, 'Technical Appendices', ha='center', va='top', 
               fontsize=20, fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def calculate_confidence_intervals(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate statistical confidence intervals for scores.
        
        Args:
            scores: List of component scores
            
        Returns:
            Confidence interval statistics
        """
        if len(scores) < 2:
            return {'error': 'Insufficient data for confidence interval calculation'}
        
        scores_array = np.array(scores)
        n = len(scores_array)
        mean = np.mean(scores_array)
        std_err = stats.sem(scores_array)
        
        # Calculate confidence interval
        confidence_interval = stats.t.interval(
            self.confidence_level, n-1, loc=mean, scale=std_err
        )
        
        return {
            'mean': round(float(mean), self.decimal_precision),
            'standard_error': round(float(std_err), self.decimal_precision),
            'confidence_level': self.confidence_level,
            'lower_bound': round(float(confidence_interval[0]), self.decimal_precision),
            'upper_bound': round(float(confidence_interval[1]), self.decimal_precision),
            'margin_of_error': round(float(confidence_interval[1] - mean), self.decimal_precision)
        }