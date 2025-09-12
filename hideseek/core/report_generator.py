import json
import csv
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import logging

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('report_generator')


class HideSeekReportGenerator:
    """
    Generates comprehensive reports and visualizations for HideSeek analysis results.
    Supports PDF, HTML, CSV, and JSON output formats.
    """
    
    def __init__(self):
        self.output_settings = config.get_output_settings()
        self.report_format = self.output_settings.get('report_format', 'pdf')
        self.include_visualizations = self.output_settings.get('include_visualizations', True)
        self.decimal_precision = self.output_settings.get('decimal_precision', 2)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        logger.info(f"ReportGenerator initialized with format: {self.report_format}")
    
    def create_test_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """
        Create comprehensive test report from analysis results.
        
        Args:
            results: Analysis results dictionary
            output_path: Optional output file path
            
        Returns:
            Path to generated report file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"hideseek_report_{timestamp}.{self.report_format}"
        
        logger.info(f"Creating {self.report_format.upper()} report: {output_path}")
        
        try:
            if self.report_format.lower() == 'pdf':
                return self._create_pdf_report(results, output_path)
            elif self.report_format.lower() == 'html':
                return self._create_html_report(results, output_path)
            elif self.report_format.lower() == 'json':
                return self._create_json_report(results, output_path)
            else:
                raise ValueError(f"Unsupported report format: {self.report_format}")
                
        except Exception as e:
            logger.error(f"Failed to create report: {str(e)}")
            raise
    
    def _create_pdf_report(self, results: Dict[str, Any], output_path: str) -> str:
        """Create PDF report with visualizations"""
        
        with PdfPages(output_path) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.9, 'HideSeek Camouflage Analysis Report', 
                   ha='center', va='center', fontsize=24, fontweight='bold')
            
            # Basic info
            test_info = results.get('test_info', {})
            ax.text(0.5, 0.8, f"Test: {test_info.get('name', 'Unknown')}", 
                   ha='center', va='center', fontsize=16)
            ax.text(0.5, 0.75, f"Date: {test_info.get('timestamp', datetime.now().isoformat())}", 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.7, f"Environment: {results.get('environment_type', 'Unknown')}", 
                   ha='center', va='center', fontsize=12)
            
            # Overall score
            overall_score = results.get('overall_score', 0)
            ax.text(0.5, 0.6, f"Overall Effectiveness Score", 
                   ha='center', va='center', fontsize=18, fontweight='bold')
            ax.text(0.5, 0.55, f"{overall_score:.1f}/100", 
                   ha='center', va='center', fontsize=32, fontweight='bold',
                   color=self._get_score_color(overall_score))
            
            # Score interpretation
            interpretation = self._interpret_score(overall_score)
            ax.text(0.5, 0.45, f"Rating: {interpretation}", 
                   ha='center', va='center', fontsize=16, 
                   color=self._get_score_color(overall_score))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Score breakdown page
            if 'component_scores' in results:
                fig = self._create_score_breakdown_chart(results['component_scores'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Radar chart
            if self.include_visualizations and 'component_scores' in results:
                fig = self._create_radar_chart(results['component_scores'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Distance effectiveness chart
            if 'distance_analysis' in results:
                fig = self._create_distance_chart(results['distance_analysis'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Detailed analysis page
            fig = self._create_detailed_analysis_page(results)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Recommendations page
            if 'recommendations' in results:
                fig = self._create_recommendations_page(results['recommendations'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        logger.info(f"PDF report created successfully: {output_path}")
        return output_path
    
    def _create_html_report(self, results: Dict[str, Any], output_path: str) -> str:
        """Create HTML report with embedded visualizations"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HideSeek Camouflage Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .score-highlight {{
            font-size: 3em;
            font-weight: bold;
            color: {self._get_score_color_hex(results.get('overall_score', 0))};
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .score-bar {{
            height: 30px;
            background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f, #2ecc71);
            border-radius: 15px;
            position: relative;
            margin: 10px 0;
        }}
        .score-marker {{
            position: absolute;
            top: -5px;
            width: 4px;
            height: 40px;
            background: black;
            border-radius: 2px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HideSeek Camouflage Analysis Report</h1>
        <h2>{results.get('test_info', {}).get('name', 'Camouflage Test')}</h2>
        <p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Environment: {results.get('environment_type', 'Unknown')}</p>
        
        <div class="score-highlight">
            Overall Score: {results.get('overall_score', 0):.1f}/100
        </div>
        <p style="font-size: 1.2em; color: {self._get_score_color_hex(results.get('overall_score', 0))};">
            <strong>{self._interpret_score(results.get('overall_score', 0))}</strong>
        </p>
    </div>
"""
        
        # Add component scores section
        if 'component_scores' in results:
            html_content += self._create_html_scores_section(results['component_scores'])
        
        # Add detailed analysis
        html_content += self._create_html_analysis_section(results)
        
        # Add visualizations if enabled
        if self.include_visualizations:
            html_content += self._create_html_charts_section(results)
        
        # Add recommendations
        if 'recommendations' in results:
            html_content += self._create_html_recommendations_section(results['recommendations'])
        
        html_content += """
</body>
</html>
"""
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report created successfully: {output_path}")
        return output_path
    
    def _create_json_report(self, results: Dict[str, Any], output_path: str) -> str:
        """Create JSON report with all data"""
        
        # Add metadata to results
        enhanced_results = {
            'metadata': {
                'report_generated': datetime.now().isoformat(),
                'hideseek_version': '1.0.0',
                'report_format': 'json'
            },
            'analysis_results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        logger.info(f"JSON report created successfully: {output_path}")
        return output_path
    
    def export_metrics_csv(self, metrics: Dict[str, Any], filepath: str) -> str:
        """
        Export metrics to CSV format.
        
        Args:
            metrics: Metrics data to export
            filepath: Output CSV file path
            
        Returns:
            Path to created CSV file
        """
        try:
            # Flatten nested dictionaries
            flattened_metrics = self._flatten_dict(metrics)
            
            # Convert to DataFrame
            df = pd.DataFrame([flattened_metrics])
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            
            logger.info(f"Metrics exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export metrics to CSV: {str(e)}")
            raise
    
    def generate_comparison_table(self, multiple_tests: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate comparison table for multiple camouflage tests.
        
        Args:
            multiple_tests: List of test result dictionaries
            
        Returns:
            DataFrame containing comparison data
        """
        try:
            comparison_data = []
            
            for i, test in enumerate(multiple_tests):
                row = {
                    'Test_ID': f"Test_{i+1}",
                    'Name': test.get('test_info', {}).get('name', f'Test {i+1}'),
                    'Environment': test.get('environment_type', 'Unknown'),
                    'Overall_Score': round(test.get('overall_score', 0), self.decimal_precision),
                    'Color_Score': round(test.get('component_scores', {}).get('color', 0), self.decimal_precision),
                    'Pattern_Score': round(test.get('component_scores', {}).get('pattern', 0), self.decimal_precision),
                    'Brightness_Score': round(test.get('component_scores', {}).get('brightness', 0), self.decimal_precision),
                    'Distance_Score': round(test.get('component_scores', {}).get('distance', 0), self.decimal_precision),
                    'Rating': self._interpret_score(test.get('overall_score', 0))
                }
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            logger.info(f"Generated comparison table with {len(multiple_tests)} tests")
            return df
            
        except Exception as e:
            logger.error(f"Failed to generate comparison table: {str(e)}")
            raise
    
    def create_visual_report(self, images: Dict[str, np.ndarray], scores: Dict[str, float], 
                           output_path: str) -> str:
        """
        Create visual report with images and scores.
        
        Args:
            images: Dictionary of images to include
            scores: Dictionary of scores
            output_path: Output file path
            
        Returns:
            Path to created visual report
        """
        try:
            num_images = len(images)
            if num_images == 0:
                raise ValueError("No images provided for visual report")
            
            # Create figure with appropriate layout
            rows = (num_images + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(16, 8 * rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (name, img) in enumerate(images.items()):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                # Display image
                if len(img.shape) == 3:
                    # Convert BGR to RGB for matplotlib
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                else:
                    ax.imshow(img, cmap='gray')
                
                # Add score text
                score = scores.get(name, 0)
                ax.set_title(f"{name}\nScore: {score:.1f}/100", 
                           fontsize=14, fontweight='bold',
                           color=self._get_score_color(score))
                ax.axis('off')
            
            # Hide empty subplots
            for idx in range(num_images, rows * 2):
                row = idx // 2
                col = idx % 2
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visual report created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create visual report: {str(e)}")
            raise
    
    # Helper methods for creating visualizations
    def _create_score_breakdown_chart(self, component_scores: Dict[str, float]) -> plt.Figure:
        """Create horizontal bar chart of component scores"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = list(component_scores.keys())
        scores = list(component_scores.values())
        colors = [self._get_score_color(score) for score in scores]
        
        bars = ax.barh(components, scores, color=colors, alpha=0.8)
        ax.set_xlabel('Score')
        ax.set_title('Component Score Breakdown', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 100)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _create_radar_chart(self, component_scores: Dict[str, float]) -> plt.Figure:
        """Create radar chart of component scores"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        components = list(component_scores.keys())
        scores = list(component_scores.values())
        
        # Number of variables
        N = len(components)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add scores
        scores += scores[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, scores, 'o-', linewidth=2, label='Camouflage Effectiveness')
        ax.fill(angles, scores, alpha=0.25)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([comp.replace('_', ' ').title() for comp in components])
        ax.set_ylim(0, 100)
        ax.set_title('Camouflage Effectiveness Radar Chart', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add concentric circles for reference
        ax.grid(True)
        
        return fig
    
    def _create_distance_chart(self, distance_analysis: Dict[str, Any]) -> plt.Figure:
        """Create distance effectiveness chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        distances = distance_analysis.get('distances', [])
        detection_probs = distance_analysis.get('detection_probabilities', [])
        
        if distances and detection_probs:
            ax.plot(distances, [100 - p*100 for p in detection_probs], 
                   marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Distance (meters)')
            ax.set_ylabel('Camouflage Effectiveness (%)')
            ax.set_title('Effectiveness vs. Distance', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add critical distance line if available
            critical_distance = distance_analysis.get('critical_distance')
            if critical_distance:
                ax.axvline(x=critical_distance, color='red', linestyle='--', alpha=0.7,
                          label=f'50% Detection Distance: {critical_distance:.1f}m')
                ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_detailed_analysis_page(self, results: Dict[str, Any]) -> plt.Figure:
        """Create detailed analysis text page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        y_pos = 0.95
        ax.text(0.5, y_pos, 'Detailed Analysis', ha='center', va='top', 
               fontsize=20, fontweight='bold')
        
        y_pos -= 0.1
        
        # Component analysis
        component_scores = results.get('component_scores', {})
        for component, score in component_scores.items():
            ax.text(0.05, y_pos, f"{component.replace('_', ' ').title()}:", 
                   ha='left', va='top', fontsize=12, fontweight='bold')
            ax.text(0.95, y_pos, f"{score:.1f}/100", 
                   ha='right', va='top', fontsize=12, 
                   color=self._get_score_color(score))
            
            # Add interpretation
            interpretation = self._get_component_interpretation(component, score)
            y_pos -= 0.03
            ax.text(0.1, y_pos, interpretation, ha='left', va='top', fontsize=10)
            y_pos -= 0.08
        
        # Environmental suitability
        if 'environment_analysis' in results:
            ax.text(0.05, y_pos, 'Environmental Suitability:', 
                   ha='left', va='top', fontsize=12, fontweight='bold')
            y_pos -= 0.05
            
            env_analysis = results['environment_analysis']
            for env, suitability in env_analysis.items():
                ax.text(0.1, y_pos, f"{env.title()}: {suitability:.1f}%", 
                       ha='left', va='top', fontsize=10)
                y_pos -= 0.03
        
        return fig
    
    def _create_recommendations_page(self, recommendations: List[str]) -> plt.Figure:
        """Create recommendations page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        y_pos = 0.95
        ax.text(0.5, y_pos, 'Recommendations', ha='center', va='top', 
               fontsize=20, fontweight='bold')
        
        y_pos -= 0.1
        
        for i, rec in enumerate(recommendations, 1):
            ax.text(0.05, y_pos, f"{i}. {rec}", ha='left', va='top', 
                   fontsize=11, wrap=True)
            y_pos -= 0.08
        
        return fig
    
    # Helper methods
    def _get_score_color(self, score: float) -> str:
        """Get matplotlib color based on score"""
        if score >= 80:
            return 'green'
        elif score >= 60:
            return 'orange'
        elif score >= 40:
            return 'gold'
        else:
            return 'red'
    
    def _get_score_color_hex(self, score: float) -> str:
        """Get hex color based on score"""
        if score >= 80:
            return '#2ecc71'
        elif score >= 60:
            return '#f39c12'
        elif score >= 40:
            return '#f1c40f'
        else:
            return '#e74c3c'
    
    def _interpret_score(self, score: float) -> str:
        """Get text interpretation of score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 50:
            return "Poor"
        else:
            return "Very Poor"
    
    def _get_component_interpretation(self, component: str, score: float) -> str:
        """Get interpretation for specific component"""
        interpretations = {
            'color': {
                90: "Excellent color matching with background",
                70: "Good color blending, minor differences",
                50: "Moderate color matching, some contrast",
                30: "Poor color matching, noticeable differences",
                0: "Very poor color matching, high contrast"
            },
            'pattern': {
                90: "Excellent pattern disruption, shape well hidden",
                70: "Good pattern breaking, some edge visibility",
                50: "Moderate pattern disruption",
                30: "Poor pattern breaking, shape recognizable",
                0: "No effective pattern disruption"
            },
            'brightness': {
                90: "Excellent brightness matching",
                70: "Good brightness levels, minor differences",
                50: "Moderate brightness matching",
                30: "Poor brightness matching, noticeable contrast",
                0: "Very poor brightness matching"
            },
            'distance': {
                90: "Effective at long distances",
                70: "Good effectiveness at medium distances",
                50: "Moderate distance effectiveness",
                30: "Poor distance effectiveness",
                0: "Easily detectable at all distances"
            }
        }
        
        comp_interpretations = interpretations.get(component, {})
        for threshold in sorted(comp_interpretations.keys(), reverse=True):
            if score >= threshold:
                return comp_interpretations[threshold]
        
        return "Analysis complete"
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _create_html_scores_section(self, component_scores: Dict[str, float]) -> str:
        """Create HTML section for component scores"""
        html = '<div class="section"><h2>Component Scores</h2>'
        html += '<table><tr><th>Component</th><th>Score</th><th>Rating</th></tr>'
        
        for component, score in component_scores.items():
            html += f"""
            <tr>
                <td>{component.replace('_', ' ').title()}</td>
                <td style="color: {self._get_score_color_hex(score)}; font-weight: bold;">
                    {score:.1f}/100
                </td>
                <td>{self._interpret_score(score)}</td>
            </tr>
            """
        
        html += '</table></div>'
        return html
    
    def _create_html_analysis_section(self, results: Dict[str, Any]) -> str:
        """Create HTML section for detailed analysis"""
        html = '<div class="section"><h2>Analysis Details</h2>'
        
        # Add key findings
        if 'key_findings' in results:
            html += '<h3>Key Findings</h3><ul>'
            for finding in results['key_findings']:
                html += f'<li>{finding}</li>'
            html += '</ul>'
        
        html += '</div>'
        return html
    
    def _create_html_charts_section(self, results: Dict[str, Any]) -> str:
        """Create HTML section with embedded charts"""
        html = '<div class="section"><h2>Visualizations</h2>'
        
        # Create charts and embed as base64
        if 'component_scores' in results:
            try:
                fig = self._create_score_breakdown_chart(results['component_scores'])
                img_data = self._fig_to_base64(fig)
                html += f'<div class="chart-container"><img src="data:image/png;base64,{img_data}" alt="Score Breakdown Chart"></div>'
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to create score chart: {e}")
        
        html += '</div>'
        return html
    
    def _create_html_recommendations_section(self, recommendations: List[str]) -> str:
        """Create HTML section for recommendations"""
        html = '<div class="section"><h2>Recommendations</h2><ol>'
        for rec in recommendations:
            html += f'<li>{rec}</li>'
        html += '</ol></div>'
        return html
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        return base64.b64encode(image_png).decode('utf-8')