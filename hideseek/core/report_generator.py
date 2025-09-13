import json
import csv
import os
import cv2
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
            
            # Image analysis page with visual explanations
            if 'original_image' in results or 'image_path' in results:
                self._create_image_analysis_page(pdf, results)
                
            # Step-by-step process explanation page
            if 'original_image' in results or 'image_path' in results:
                self._create_process_explanation_page(pdf, results)
            
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
    
    def _create_image_analysis_page(self, pdf, results: Dict[str, Any]):
        """Create a page with image analysis and visual explanations"""
        
        # Load the original image
        image_path = results.get('image_path')
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif 'original_image' in results:
            image_rgb = results['original_image']
        else:
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(8.5, 11))
        
        # Main title
        fig.suptitle('Visual Analysis Breakdown', fontsize=20, fontweight='bold', y=0.95)
        
        # Original image (top half)
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Color analysis visualization (top right)
        ax2 = plt.subplot(2, 2, 2)
        # Get color analysis from the correct nested location
        color_analysis = {}
        if 'detailed_results' in results and 'color_blending' in results['detailed_results']:
            color_analysis = results['detailed_results']['color_blending'].get('color_analysis', {})
        self._create_color_analysis_visual(ax2, color_analysis, image_rgb)
        
        # Object segmentation mask (bottom left)
        ax3 = plt.subplot(2, 2, 3)
        self._create_segmentation_visual(ax3, results.get('segmentation', {}), image_rgb)
        
        # Score summary with visual indicators (bottom right)
        ax4 = plt.subplot(2, 2, 4)
        self._create_score_visual_summary(ax4, results)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_color_analysis_visual(self, ax, color_analysis: Dict[str, Any], image: np.ndarray):
        """Create color analysis visualization"""
        ax.set_title('Color Blending Analysis', fontsize=12, fontweight='bold')
        
        if not color_analysis:
            ax.text(0.5, 0.5, 'Color analysis not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Show color analysis statistics
        if 'delta_e_statistics' in color_analysis:
            stats = color_analysis['delta_e_statistics']
            # Create a visualization of color matching quality
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            
            # Title and main score
            score = color_analysis.get('color_matching_score', 0)
            ax.text(5, 9, f'Color Matching: {score:.1f}/100', 
                   ha='center', fontsize=12, fontweight='bold')
            
            # Delta-E statistics
            mean_de = stats.get('mean_delta_e', 0)
            ax.text(5, 7.5, f'Avg Color Difference: {mean_de:.1f} ΔE', 
                   ha='center', fontsize=10)
            
            # Quality indicators
            if 'quality_distribution' in color_analysis:
                quality = color_analysis['quality_distribution']
                excellent = quality.get('excellent', 0)
                good = quality.get('good', 0) - excellent
                acceptable = quality.get('acceptable', 0) - quality.get('good', 0)
                
                ax.text(5, 6, f'Quality Distribution:', ha='center', fontsize=10, fontweight='bold')
                ax.text(5, 5.2, f'Excellent: {excellent}%', ha='center', fontsize=9, color='green')
                ax.text(5, 4.6, f'Good: {good}%', ha='center', fontsize=9, color='blue')
                ax.text(5, 4.0, f'Acceptable: {acceptable}%', ha='center', fontsize=9, color='orange')
            
            # Color scale reference
            ax.text(5, 2.5, 'ΔE Scale:\n0-2: Excellent\n2-5: Good\n5-10: Acceptable\n>10: Poor', 
                   ha='center', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        else:
            # Show a small version of the image with color analysis overlay
            small_img = cv2.resize(image, (200, 150))
            ax.imshow(small_img)
            
            # Add color score overlay
            score = color_analysis.get('color_matching_score', 0)
            ax.text(10, 20, f'Color Score: {score:.1f}/100', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        ax.axis('off')
    
    def _create_segmentation_visual(self, ax, segmentation: Dict[str, Any], image: np.ndarray):
        """Create object segmentation visualization"""
        ax.set_title('Object Detection', fontsize=12, fontweight='bold')
        
        if 'object_mask' in segmentation:
            mask = segmentation['object_mask']
            # Create overlay
            overlay = image.copy()
            overlay[mask > 0] = [255, 255, 0]  # Highlight object in yellow
            result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            ax.imshow(result)
        else:
            # Show original image with bounding box if available
            display_img = image.copy()
            if 'bounding_box' in segmentation:
                x, y, w, h = segmentation['bounding_box']
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            ax.imshow(display_img)
            ax.text(10, 20, 'Object segmentation applied', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
        
        ax.axis('off')
    
    def _create_score_visual_summary(self, ax, results: Dict[str, Any]):
        """Create visual score summary with color-coded indicators"""
        ax.set_title('Performance Indicators', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Get component scores
        components = []
        if 'component_scores' in results:
            for name, score in results['component_scores'].items():
                components.append((name.replace('_', ' ').title(), score))
        
        # If no component scores, use overall score
        if not components:
            overall = results.get('overall_score', 0)
            components = [('Overall', overall)]
        
        # Draw score indicators
        y_pos = 8.5
        for name, score in components[:5]:  # Show top 5 components
            # Score bar
            bar_width = score / 100 * 7  # Scale to fit
            color = self._get_score_color(score)
            
            # Background bar
            ax.add_patch(plt.Rectangle((2, y_pos-0.2), 7, 0.4, 
                                     facecolor='lightgray', alpha=0.3))
            # Score bar
            ax.add_patch(plt.Rectangle((2, y_pos-0.2), bar_width, 0.4, 
                                     facecolor=color, alpha=0.8))
            
            # Label and score text
            ax.text(0.5, y_pos, name, fontsize=10, va='center', ha='left')
            ax.text(9.5, y_pos, f'{score:.1f}', fontsize=10, va='center', ha='right')
            
            y_pos -= 1.2
        
        # Add legend
        ax.text(5, 1, 'Performance Scale:\n90-100: Excellent\n70-89: Good\n50-69: Fair\n<50: Poor', 
               ha='center', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    def _create_process_explanation_page(self, pdf, results: Dict[str, Any]):
        """Create a detailed visual explanation of the analysis process"""
        
        # Load the original image
        image_path = results.get('image_path')
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif 'original_image' in results:
            image_rgb = results['original_image']
        else:
            return
            
        # Create figure with 6 subplots (2x3 grid)
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('🔍 How HideSeek Analyzes Camouflage - Step by Step', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Step 1: Original Image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image_rgb)
        ax1.set_title('Step 1: Original Image\n📸 What we start with', fontsize=10, fontweight='bold')
        ax1.text(0.5, -0.15, 'This is the camouflage photo\nwe want to analyze', 
                ha='center', va='top', transform=ax1.transAxes, fontsize=8)
        ax1.axis('off')
        
        # Step 2: Environment Detection
        ax2 = plt.subplot(2, 3, 2)
        self._create_environment_detection_visual(ax2, results, image_rgb)
        
        # Step 3: Object Detection
        ax3 = plt.subplot(2, 3, 3)
        self._create_object_detection_visual(ax3, results, image_rgb)
        
        # Step 4: Color Analysis
        ax4 = plt.subplot(2, 3, 4)
        self._create_color_process_visual(ax4, results, image_rgb)
        
        # Step 5: Background Sampling
        ax5 = plt.subplot(2, 3, 5)
        self._create_background_sampling_visual(ax5, results, image_rgb)
        
        # Step 6: Final Scoring
        ax6 = plt.subplot(2, 3, 6)
        self._create_scoring_process_visual(ax6, results)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.4)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_environment_detection_visual(self, ax, results: Dict[str, Any], image: np.ndarray):
        """Visual explanation of environment detection"""
        ax.set_title('Step 2: Environment Detection\n🌲 AI identifies the setting', fontsize=10, fontweight='bold')
        
        # Show small image with environment overlay
        small_img = cv2.resize(image, (150, 100))
        ax.imshow(small_img)
        
        # Get environment info
        env_type = results.get('environment_type', 'unknown')
        env_confidence = results.get('environment_confidence', 0)
        
        # Add environment label
        env_colors = {
            'woodland': 'green',
            'desert': 'orange', 
            'urban': 'gray',
            'arctic': 'lightblue',
            'tropical': 'lime'
        }
        color = env_colors.get(env_type, 'black')
        
        ax.text(75, 15, f'{env_type.upper()}\n{env_confidence:.0%} confident', 
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        ax.text(0.5, -0.15, f'AI analyzes colors & textures\nto identify: {env_type}', 
                ha='center', va='top', transform=ax.transAxes, fontsize=8)
        ax.axis('off')
    
    def _create_object_detection_visual(self, ax, results: Dict[str, Any], image: np.ndarray):
        """Visual explanation of automatic object detection"""
        ax.set_title('Step 3: Object Detection\n🎯 AI finds the camouflaged object', fontsize=10, fontweight='bold')
        
        # Create a simulated segmentation mask for visualization
        h, w = image.shape[:2]
        small_img = cv2.resize(image, (150, 100))
        h_small, w_small = small_img.shape[:2]
        
        # Create a center-focused mask to show the concept
        center_y, center_x = h_small // 2, w_small // 2
        mask = np.zeros((h_small, w_small), dtype=np.uint8)
        
        # Create an oval-shaped object detection area
        y_indices, x_indices = np.ogrid[:h_small, :w_small]
        mask_area = ((x_indices - center_x) ** 2 / (w_small // 4) ** 2 + 
                     (y_indices - center_y) ** 2 / (h_small // 6) ** 2) <= 1
        mask[mask_area] = 255
        
        # Create overlay showing detected object
        overlay = small_img.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([255, 255, 0]) * 0.3
        ax.imshow(overlay)
        
        # Add outline around detected area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.reshape(-1, 2)
            ax.plot(contour[:, 0], contour[:, 1], 'red', linewidth=2)
        
        ax.text(0.5, -0.15, 'K-means clustering finds\nthe object automatically', 
                ha='center', va='top', transform=ax.transAxes, fontsize=8)
        ax.axis('off')
    
    def _create_color_process_visual(self, ax, results: Dict[str, Any], image: np.ndarray):
        """Visual explanation of color analysis process"""
        ax.set_title('Step 4: Color Analysis\n🎨 Measure color matching', fontsize=10, fontweight='bold')
        
        # Get color analysis data
        color_analysis = {}
        if 'detailed_results' in results and 'color_blending' in results['detailed_results']:
            color_analysis = results['detailed_results']['color_blending'].get('color_analysis', {})
        
        # Create color comparison visualization
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Object colors (left side)
        ax.add_patch(plt.Rectangle((1, 6), 1.5, 2, facecolor='brown', alpha=0.7))
        ax.text(1.75, 7, 'Object\nColors', ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Background colors (right side)  
        ax.add_patch(plt.Rectangle((7.5, 6), 1.5, 2, facecolor='green', alpha=0.7))
        ax.text(8.25, 7, 'Background\nColors', ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Arrow showing comparison
        ax.annotate('', xy=(7.5, 7), xytext=(2.5, 7),
                   arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
        ax.text(5, 7.5, 'Compare', ha='center', fontsize=9, fontweight='bold', color='red')
        
        # Show Delta-E result if available
        if 'delta_e_statistics' in color_analysis:
            mean_de = color_analysis['delta_e_statistics'].get('mean_delta_e', 0)
            ax.text(5, 5, f'Average Color Difference:\n{mean_de:.1f} ΔE units', 
                   ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        ax.text(5, 2, 'Uses scientific color formulas\n(CIEDE2000) to measure\nhow well colors match', 
                ha='center', fontsize=8)
    
    def _create_background_sampling_visual(self, ax, results: Dict[str, Any], image: np.ndarray):
        """Visual explanation of background sampling"""
        ax.set_title('Step 5: Background Sampling\n📊 Collect reference colors', fontsize=10, fontweight='bold')
        
        # Show image with background ring visualization
        small_img = cv2.resize(image, (150, 100))
        ax.imshow(small_img)
        
        h, w = small_img.shape[:2]
        
        # Draw background sampling ring (outer area)
        ring_inner = min(h, w) // 4
        ring_outer = min(h, w) // 2.5
        
        theta = np.linspace(0, 2*np.pi, 100)
        center_x, center_y = w // 2, h // 2
        
        # Outer ring
        x_outer = center_x + ring_outer * np.cos(theta)
        y_outer = center_y + ring_outer * np.sin(theta)
        ax.plot(x_outer, y_outer, 'blue', linewidth=3, alpha=0.8)
        
        # Inner boundary
        x_inner = center_x + ring_inner * np.cos(theta) 
        y_inner = center_y + ring_inner * np.sin(theta)
        ax.plot(x_inner, y_inner, 'blue', linewidth=2, alpha=0.8, linestyle='--')
        
        # Add sample points
        for i in range(8):
            angle = i * np.pi / 4
            x_sample = center_x + (ring_outer - 5) * np.cos(angle)
            y_sample = center_y + (ring_outer - 5) * np.sin(angle)
            ax.plot(x_sample, y_sample, 'bo', markersize=4)
        
        ax.text(0.5, -0.15, 'Creates ring around object\nto sample background colors', 
                ha='center', va='top', transform=ax.transAxes, fontsize=8)
        ax.axis('off')
    
    def _create_scoring_process_visual(self, ax, results: Dict[str, Any]):
        """Visual explanation of the scoring process"""
        ax.set_title('Step 6: Final Score\n📈 Calculate effectiveness', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Get scores
        overall_score = results.get('overall_score', 0)
        component_scores = results.get('component_scores', {})
        
        # Create simple score visualization
        ax.text(5, 8.5, f'OVERALL SCORE', ha='center', fontsize=12, fontweight='bold')
        ax.text(5, 7.5, f'{overall_score:.1f}/100', ha='center', fontsize=20, fontweight='bold',
               color=self._get_score_color(overall_score))
        
        # Show key components
        y_pos = 6
        for name, score in list(component_scores.items())[:3]:
            display_name = name.replace('_', ' ').title()
            ax.text(5, y_pos, f'{display_name}: {score:.1f}/100', 
                   ha='center', fontsize=9)
            y_pos -= 0.8
        
        # Explanation
        ax.text(5, 2.5, 'Combines all analysis results\ninto final camouflage\neffectiveness rating', 
                ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # Rating interpretation
        if overall_score >= 80:
            rating = "Excellent camouflage! 🌟"
        elif overall_score >= 60:
            rating = "Good camouflage ✅"
        elif overall_score >= 40:
            rating = "Fair camouflage ⚠️"
        else:
            rating = "Poor camouflage ❌"
            
        ax.text(5, 1, rating, ha='center', fontsize=10, fontweight='bold')
    
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