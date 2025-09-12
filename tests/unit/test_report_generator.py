import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from hideseek.core.report_generator import HideSeekReportGenerator


class TestHideSeekReportGenerator:
    """Test suite for HideSeekReportGenerator"""
    
    @pytest.fixture
    def report_generator(self):
        """Create ReportGenerator instance for testing"""
        return HideSeekReportGenerator()
    
    @pytest.fixture
    def sample_results(self):
        """Create sample analysis results for testing"""
        return {
            'test_info': {
                'name': 'Test Camouflage Pattern',
                'timestamp': '2024-01-15T10:30:00'
            },
            'environment_type': 'woodland',
            'overall_score': 76.5,
            'component_scores': {
                'color': 78.2,
                'pattern': 74.5,
                'brightness': 80.1,
                'distance': 73.0
            },
            'distance_analysis': {
                'distances': [5, 10, 25, 50, 100],
                'detection_probabilities': [0.1, 0.2, 0.4, 0.7, 0.9],
                'critical_distance': 35.5
            },
            'environment_analysis': {
                'woodland': 85.0,
                'desert': 45.2,
                'urban': 62.8
            },
            'key_findings': [
                'Good color matching with woodland environment',
                'Pattern disruption effective at medium distances',
                'Brightness levels well matched to environment'
            ],
            'recommendations': [
                'Consider darker colors for better shadow matching',
                'Add more vertical pattern elements',
                'Test performance in different lighting conditions'
            ]
        }
    
    def test_init(self, report_generator):
        """Test ReportGenerator initialization"""
        assert report_generator is not None
        assert hasattr(report_generator, 'report_format')
        assert hasattr(report_generator, 'include_visualizations')
        assert hasattr(report_generator, 'decimal_precision')
    
    def test_create_json_report(self, report_generator, sample_results):
        """Test JSON report creation"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result_path = report_generator._create_json_report(sample_results, output_path)
            
            assert result_path == output_path
            assert Path(output_path).exists()
            
            # Check content
            with open(output_path, 'r') as f:
                report_data = json.load(f)
            
            assert 'metadata' in report_data
            assert 'analysis_results' in report_data
            assert report_data['analysis_results'] == sample_results
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_pdf_report(self, mock_savefig, report_generator, sample_results):
        """Test PDF report creation (mocked)"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Mock PdfPages context manager
            with patch('matplotlib.backends.backend_pdf.PdfPages') as mock_pdf:
                mock_pdf.return_value.__enter__ = MagicMock()
                mock_pdf.return_value.__exit__ = MagicMock()
                
                result_path = report_generator._create_pdf_report(sample_results, output_path)
                assert result_path == output_path
                
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_create_html_report(self, report_generator, sample_results):
        """Test HTML report creation"""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result_path = report_generator._create_html_report(sample_results, output_path)
            
            assert result_path == output_path
            assert Path(output_path).exists()
            
            # Check basic HTML structure
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            assert '<!DOCTYPE html>' in html_content
            assert 'HideSeek Camouflage Analysis Report' in html_content
            assert str(sample_results['overall_score']) in html_content
            assert sample_results['environment_type'] in html_content
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_export_metrics_csv(self, report_generator):
        """Test CSV export functionality"""
        metrics = {
            'overall_score': 75.5,
            'component_scores': {
                'color': 78.0,
                'pattern': 73.0
            },
            'environment': 'woodland'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result_path = report_generator.export_metrics_csv(metrics, output_path)
            
            assert result_path == output_path
            assert Path(output_path).exists()
            
            # Read and verify CSV content
            df = pd.read_csv(output_path)
            assert len(df) == 1
            assert 'overall_score' in df.columns
            assert 'component_scores_color' in df.columns
            assert 'component_scores_pattern' in df.columns
            assert 'environment' in df.columns
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_generate_comparison_table(self, report_generator):
        """Test comparison table generation"""
        test_results = [
            {
                'test_info': {'name': 'Test 1'},
                'environment_type': 'woodland',
                'overall_score': 75.0,
                'component_scores': {'color': 80, 'pattern': 70, 'brightness': 75, 'distance': 75}
            },
            {
                'test_info': {'name': 'Test 2'},
                'environment_type': 'desert',
                'overall_score': 82.5,
                'component_scores': {'color': 85, 'pattern': 80, 'brightness': 82, 'distance': 83}
            }
        ]
        
        comparison_df = report_generator.generate_comparison_table(test_results)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        
        # Check columns
        expected_columns = ['Test_ID', 'Name', 'Environment', 'Overall_Score', 
                          'Color_Score', 'Pattern_Score', 'Brightness_Score', 
                          'Distance_Score', 'Rating']
        for col in expected_columns:
            assert col in comparison_df.columns
        
        # Check data
        assert comparison_df.loc[0, 'Name'] == 'Test 1'
        assert comparison_df.loc[1, 'Name'] == 'Test 2'
        assert comparison_df.loc[0, 'Overall_Score'] == 75.0
        assert comparison_df.loc[1, 'Overall_Score'] == 82.5
    
    def test_score_color_mapping(self, report_generator):
        """Test score color mapping functions"""
        # Test matplotlib colors
        assert report_generator._get_score_color(95) == 'green'
        assert report_generator._get_score_color(75) == 'orange' 
        assert report_generator._get_score_color(55) == 'gold'
        assert report_generator._get_score_color(25) == 'red'
        
        # Test hex colors
        assert report_generator._get_score_color_hex(95) == '#2ecc71'
        assert report_generator._get_score_color_hex(75) == '#f39c12'
        assert report_generator._get_score_color_hex(55) == '#f1c40f'
        assert report_generator._get_score_color_hex(25) == '#e74c3c'
    
    def test_score_interpretation(self, report_generator):
        """Test score interpretation function"""
        assert report_generator._interpret_score(95) == "Excellent"
        assert report_generator._interpret_score(85) == "Very Good"
        assert report_generator._interpret_score(75) == "Good"
        assert report_generator._interpret_score(65) == "Fair"
        assert report_generator._interpret_score(55) == "Poor"
        assert report_generator._interpret_score(25) == "Very Poor"
    
    def test_component_interpretation(self, report_generator):
        """Test component-specific interpretation"""
        color_interp = report_generator._get_component_interpretation('color', 85)
        assert 'color' in color_interp.lower()
        
        pattern_interp = report_generator._get_component_interpretation('pattern', 65)
        assert 'pattern' in pattern_interp.lower()
        
        brightness_interp = report_generator._get_component_interpretation('brightness', 45)
        assert 'brightness' in brightness_interp.lower()
        
        distance_interp = report_generator._get_component_interpretation('distance', 90)
        assert 'distance' in distance_interp.lower()
    
    def test_flatten_dict(self, report_generator):
        """Test dictionary flattening function"""
        nested_dict = {
            'level1': {
                'level2': {
                    'value': 42
                },
                'simple': 'test'
            },
            'root': 'root_value'
        }
        
        flattened = report_generator._flatten_dict(nested_dict)
        
        assert 'level1_level2_value' in flattened
        assert 'level1_simple' in flattened
        assert 'root' in flattened
        
        assert flattened['level1_level2_value'] == 42
        assert flattened['level1_simple'] == 'test'
        assert flattened['root'] == 'root_value'
    
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.savefig')
    def test_create_visual_report(self, mock_savefig, mock_close, report_generator):
        """Test visual report creation with mocked matplotlib"""
        images = {
            'original': np.zeros((100, 100, 3), dtype=np.uint8),
            'processed': np.ones((100, 100, 3), dtype=np.uint8) * 128
        }
        scores = {
            'original': 65.0,
            'processed': 78.5
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Mock cv2.cvtColor to avoid import issues in testing
            with patch('hideseek.core.report_generator.cv2.cvtColor') as mock_cvt:
                mock_cvt.return_value = images['original']
                
                result_path = report_generator.create_visual_report(
                    images, scores, output_path
                )
                
                assert result_path == output_path
                mock_savefig.assert_called_once()
                mock_close.assert_called_once()
                
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_create_visual_report_no_images(self, report_generator):
        """Test visual report creation with no images"""
        with pytest.raises(ValueError, match="No images provided"):
            report_generator.create_visual_report({}, {}, 'output.png')
    
    def test_create_test_report_unsupported_format(self, report_generator, sample_results):
        """Test creating report with unsupported format"""
        # Temporarily change format to unsupported
        original_format = report_generator.report_format
        report_generator.report_format = 'unsupported'
        
        try:
            with pytest.raises(ValueError, match="Unsupported report format"):
                report_generator.create_test_report(sample_results, 'output.unsupported')
        finally:
            report_generator.report_format = original_format
    
    @patch('matplotlib.pyplot.subplots')
    def test_create_score_breakdown_chart(self, mock_subplots, report_generator):
        """Test score breakdown chart creation"""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        component_scores = {
            'color': 78.2,
            'pattern': 74.5,
            'brightness': 80.1,
            'distance': 73.0
        }
        
        fig = report_generator._create_score_breakdown_chart(component_scores)
        
        assert fig == mock_fig
        mock_ax.barh.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_title.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    def test_create_radar_chart(self, mock_subplots, report_generator):
        """Test radar chart creation"""
        # Mock matplotlib components with polar projection
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        component_scores = {
            'color': 78.2,
            'pattern': 74.5,
            'brightness': 80.1,
            'distance': 73.0
        }
        
        fig = report_generator._create_radar_chart(component_scores)
        
        assert fig == mock_fig
        mock_ax.plot.assert_called_once()
        mock_ax.fill.assert_called_once()
        mock_ax.set_title.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])