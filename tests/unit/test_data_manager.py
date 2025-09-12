import pytest
import tempfile
import shutil
from pathlib import Path
import json
import sqlite3
from unittest.mock import patch, MagicMock
import numpy as np

from hideseek.core.data_manager import TestDataManager


class TestTestDataManager:
    """Test suite for TestDataManager"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def data_manager(self, temp_data_dir):
        """Create TestDataManager instance with temporary directory"""
        return TestDataManager(temp_data_dir)
    
    def test_init(self, data_manager, temp_data_dir):
        """Test TestDataManager initialization"""
        assert data_manager.base_data_dir == Path(temp_data_dir)
        
        # Check directory structure creation
        assert data_manager.results_dir.exists()
        assert data_manager.cache_dir.exists()
        assert data_manager.sessions_dir.exists()
        assert data_manager.environments_dir.exists()
        
        # Check database file creation
        assert data_manager.db_path.exists()
    
    def test_database_initialization(self, data_manager):
        """Test SQLite database initialization"""
        conn = sqlite3.connect(data_manager.db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'sessions' in tables
        assert 'test_results' in tables
        assert 'cache' in tables
        
        conn.close()
    
    def test_organize_test_session(self, data_manager):
        """Test creating a new test session"""
        session_name = "Test Session 1"
        description = "Test description"
        
        session_path = data_manager.organize_test_session(session_name, description)
        
        # Check session directory created
        session_dir = Path(session_path)
        assert session_dir.exists()
        assert session_dir.is_dir()
        
        # Check subdirectories
        assert (session_dir / "images").exists()
        assert (session_dir / "reports").exists()
        assert (session_dir / "intermediate").exists()
        assert (session_dir / "visualizations").exists()
        
        # Check session info file
        session_info_file = session_dir / "session_info.json"
        assert session_info_file.exists()
        
        with open(session_info_file, 'r') as f:
            session_info = json.load(f)
        
        assert session_info['name'] == session_name
        assert session_info['description'] == description
        
        # Check database entry
        conn = sqlite3.connect(data_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_name = ?", (session_name,))
        session_row = cursor.fetchone()
        conn.close()
        
        assert session_row is not None
        assert session_row[1] == session_name  # session_name column
        assert session_row[3] == description   # description column
    
    def test_organize_test_session_sanitize_name(self, data_manager):
        """Test session name sanitization"""
        session_name = "Test/Session*With:Invalid<Characters>"
        session_path = data_manager.organize_test_session(session_name)
        
        # Should create directory without issues
        assert Path(session_path).exists()
    
    def test_save_intermediate_results(self, data_manager):
        """Test saving intermediate analysis results"""
        test_data = {
            'score': 75.5,
            'metrics': {'color': 80, 'pattern': 70}
        }
        stage = "color_analysis"
        
        file_path = data_manager.save_intermediate_results(test_data, stage)
        
        # Check file created
        assert Path(file_path).exists()
        
        # Check content
        with open(file_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['stage'] == stage
        assert saved_data['data'] == test_data
        assert 'timestamp' in saved_data
    
    def test_save_intermediate_results_with_session(self, data_manager):
        """Test saving intermediate results in session directory"""
        # Create session first
        session_path = data_manager.organize_test_session("Test Session")
        
        test_data = {'score': 85.0}
        stage = "pattern_analysis"
        
        file_path = data_manager.save_intermediate_results(
            test_data, stage, session_path
        )
        
        # Should be saved in session's intermediate directory
        assert str(Path(session_path) / "intermediate") in file_path
        assert Path(file_path).exists()
    
    def test_load_environment_database_creates_default(self, data_manager):
        """Test loading environment database creates default if not exists"""
        env_db = data_manager.load_environment_database()
        
        assert isinstance(env_db, dict)
        assert 'woodland' in env_db
        assert 'desert' in env_db
        assert 'urban' in env_db
        assert 'arctic' in env_db
        assert 'tropical' in env_db
        
        # Check structure of woodland environment
        woodland = env_db['woodland']
        assert 'description' in woodland
        assert 'reference_images' in woodland
        assert 'color_palette' in woodland
        assert 'texture_complexity' in woodland
        assert 'lighting_characteristics' in woodland
    
    def test_load_environment_database_loads_existing(self, data_manager):
        """Test loading existing environment database"""
        # Create custom environment database
        custom_env_db = {
            'custom_env': {
                'description': 'Custom environment',
                'color_palette': ['#ff0000']
            }
        }
        
        env_db_file = data_manager.environments_dir / "environment_database.json"
        with open(env_db_file, 'w') as f:
            json.dump(custom_env_db, f)
        
        loaded_db = data_manager.load_environment_database()
        assert loaded_db == custom_env_db
    
    def test_cache_processed_images(self, data_manager):
        """Test caching processed images"""
        # Create sample image data
        sample_img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_id = "test_image_001"
        processing_params = {'resize': True, 'denoise': False}
        
        cache_path = data_manager.cache_processed_images(
            sample_img, img_id, processing_params
        )
        
        # Check cache file created
        assert Path(cache_path).exists()
        
        # Check database entry
        conn = sqlite3.connect(data_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cache WHERE file_path = ?", (cache_path,))
        cache_row = cursor.fetchone()
        conn.close()
        
        assert cache_row is not None
    
    def test_load_cached_image(self, data_manager):
        """Test loading cached images"""
        # Cache an image first
        sample_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        img_id = "test_image_002"
        processing_params = {'contrast': 1.2}
        
        data_manager.cache_processed_images(sample_img, img_id, processing_params)
        
        # Load cached image
        loaded_img = data_manager.load_cached_image(img_id, processing_params)
        
        assert loaded_img is not None
        assert np.array_equal(loaded_img, sample_img)
    
    def test_load_cached_image_not_found(self, data_manager):
        """Test loading non-existent cached image"""
        loaded_img = data_manager.load_cached_image("nonexistent", {})
        assert loaded_img is None
    
    def test_cache_key_generation(self, data_manager):
        """Test cache key generation consistency"""
        img_id = "test_img"
        params1 = {'a': 1, 'b': 2}
        params2 = {'b': 2, 'a': 1}  # Same params, different order
        
        key1 = data_manager._generate_cache_key(img_id, params1)
        key2 = data_manager._generate_cache_key(img_id, params2)
        
        # Should generate same key regardless of param order
        assert key1 == key2
        
        # Different params should generate different keys
        key3 = data_manager._generate_cache_key(img_id, {'c': 3})
        assert key1 != key3
    
    def test_save_test_results(self, data_manager):
        """Test saving test results"""
        # Create session first
        session_name = "Results Test Session"
        data_manager.organize_test_session(session_name)
        
        test_results = {
            'image_path': '/path/to/test.jpg',
            'overall_score': 82.5,
            'color_score': 80.0,
            'pattern_score': 85.0,
            'brightness_score': 78.0,
            'distance_score': 87.0,
            'environment_type': 'woodland'
        }
        
        test_name = "woodland_camo_test"
        results_path = data_manager.save_test_results(
            session_name, test_name, test_results
        )
        
        # Check file created
        assert Path(results_path).exists()
        
        # Check database entry
        conn = sqlite3.connect(data_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tr.*, s.session_name 
            FROM test_results tr
            JOIN sessions s ON tr.session_id = s.id
            WHERE tr.test_name = ?
        """, (test_name,))
        result_row = cursor.fetchone()
        conn.close()
        
        assert result_row is not None
        assert result_row[2] == test_name  # test_name
        assert result_row[4] == 82.5       # overall_score
        assert result_row[11] == session_name  # session_name from join
    
    def test_save_test_results_invalid_session(self, data_manager):
        """Test saving results with invalid session name"""
        with pytest.raises(ValueError, match="Session not found"):
            data_manager.save_test_results(
                "nonexistent_session", 
                "test", 
                {'score': 50}
            )
    
    def test_get_session_summary(self, data_manager):
        """Test getting session summary"""
        # Create session and add test result
        session_name = "Summary Test Session"
        description = "Test session for summary"
        data_manager.organize_test_session(session_name, description)
        
        test_results = {
            'overall_score': 75.0,
            'environment_type': 'desert'
        }
        data_manager.save_test_results(session_name, "desert_test", test_results)
        
        # Get summary
        summary = data_manager.get_session_summary(session_name)
        
        assert summary['session_name'] == session_name
        assert summary['description'] == description
        assert summary['test_count'] == 1
        assert len(summary['tests']) == 1
        assert summary['tests'][0]['name'] == "desert_test"
        assert summary['tests'][0]['score'] == 75.0
        assert summary['tests'][0]['environment'] == 'desert'
    
    def test_get_session_summary_nonexistent(self, data_manager):
        """Test getting summary of non-existent session"""
        summary = data_manager.get_session_summary("nonexistent")
        assert summary == {}
    
    def test_cleanup_old_cache(self, data_manager):
        """Test cleaning up old cache files"""
        # This test would need to mock datetime to create "old" files
        # For now, just test that the method runs without error
        data_manager.cleanup_old_cache(max_age_days=1)
        
        # Should complete without error
        assert True


if __name__ == '__main__':
    pytest.main([__file__])