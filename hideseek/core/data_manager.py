import json
import pickle
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import shutil
import hashlib
import logging

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('data_manager')


class TestDataManager:
    """
    Manages test sessions, intermediate results, and data persistence for HideSeek.
    Handles organization of test data, caching, and database operations.
    """
    
    def __init__(self, base_data_dir: str = None):
        if base_data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.base_data_dir = project_root / "data"
        else:
            self.base_data_dir = Path(base_data_dir)
        
        # Create directory structure
        self.results_dir = self.base_data_dir / "results"
        self.cache_dir = self.base_data_dir / "cache" 
        self.sessions_dir = self.results_dir / "sessions"
        self.environments_dir = self.base_data_dir / "environments"
        
        # Create directories if they don't exist
        for directory in [self.results_dir, self.cache_dir, self.sessions_dir, self.environments_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Database file for metadata
        self.db_path = self.base_data_dir / "hideseek_data.db"
        self._init_database()
        
        logger.info(f"TestDataManager initialized with base directory: {self.base_data_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    configuration TEXT,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Create test_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    test_name TEXT NOT NULL,
                    image_path TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overall_score REAL,
                    color_score REAL,
                    pattern_score REAL,
                    brightness_score REAL,
                    distance_score REAL,
                    environment_type TEXT,
                    results_file TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    file_size INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.debug("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def organize_test_session(self, session_name: str, description: str = None) -> str:
        """
        Create and organize a new test session.
        
        Args:
            session_name: Name for the test session
            description: Optional description
            
        Returns:
            Path to the session directory
        """
        # Sanitize session name
        safe_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        session_dir = self.sessions_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        
        try:
            # Create session directory structure
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "images").mkdir(exist_ok=True)
            (session_dir / "reports").mkdir(exist_ok=True)
            (session_dir / "intermediate").mkdir(exist_ok=True)
            (session_dir / "visualizations").mkdir(exist_ok=True)
            
            # Store session info in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_config = json.dumps(config._config, indent=2)
            
            cursor.execute('''
                INSERT INTO sessions (session_name, description, configuration)
                VALUES (?, ?, ?)
            ''', (session_name, description, current_config))
            
            conn.commit()
            conn.close()
            
            # Create session info file
            session_info = {
                'name': session_name,
                'description': description,
                'created': datetime.now().isoformat(),
                'directory': str(session_dir),
                'configuration': config._config
            }
            
            with open(session_dir / "session_info.json", 'w') as f:
                json.dump(session_info, f, indent=2)
            
            logger.info(f"Created test session: {session_name}")
            logger.info(f"Session directory: {session_dir}")
            
            return str(session_dir)
            
        except Exception as e:
            logger.error(f"Failed to create session {session_name}: {str(e)}")
            raise
    
    def save_intermediate_results(self, data: Dict[str, Any], stage: str, session_dir: str = None) -> str:
        """
        Save intermediate analysis results.
        
        Args:
            data: Analysis data to save
            stage: Analysis stage name
            session_dir: Session directory (optional)
            
        Returns:
            Path to saved file
        """
        if session_dir is None:
            # Save to general intermediate directory
            save_dir = self.results_dir / "intermediate"
        else:
            save_dir = Path(session_dir) / "intermediate"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{stage}_{timestamp}.json"
        file_path = save_dir / filename
        
        try:
            # Add metadata
            save_data = {
                'stage': stage,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.debug(f"Saved intermediate results: {filename}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}")
            raise
    
    def load_environment_database(self) -> Dict[str, Any]:
        """
        Load environment reference database.
        
        Returns:
            Dictionary containing environment data
        """
        env_db_file = self.environments_dir / "environment_database.json"
        
        # Create default environment database if it doesn't exist
        if not env_db_file.exists():
            default_env_db = self._create_default_environment_database()
            with open(env_db_file, 'w') as f:
                json.dump(default_env_db, f, indent=2)
            logger.info("Created default environment database")
        
        try:
            with open(env_db_file, 'r') as f:
                env_database = json.load(f)
            logger.debug("Loaded environment database")
            return env_database
            
        except Exception as e:
            logger.error(f"Failed to load environment database: {str(e)}")
            return {}
    
    def _create_default_environment_database(self) -> Dict[str, Any]:
        """Create default environment reference database"""
        return {
            'woodland': {
                'description': 'Forest and woodland environments',
                'reference_images': [],
                'color_palette': ['#2d4a2b', '#4a6b3a', '#8b7355', '#654321'],
                'texture_complexity': 0.8,
                'lighting_characteristics': ['filtered', 'dappled', 'low_contrast']
            },
            'desert': {
                'description': 'Arid desert environments',
                'reference_images': [],
                'color_palette': ['#c19a6b', '#daa520', '#f4a460', '#cd853f'],
                'texture_complexity': 0.3,
                'lighting_characteristics': ['harsh', 'bright', 'high_contrast']
            },
            'urban': {
                'description': 'Urban and industrial environments',
                'reference_images': [],
                'color_palette': ['#696969', '#708090', '#2f4f4f', '#000000'],
                'texture_complexity': 0.9,
                'lighting_characteristics': ['artificial', 'mixed', 'shadow_heavy']
            },
            'arctic': {
                'description': 'Snow and ice environments',
                'reference_images': [],
                'color_palette': ['#ffffff', '#f0f8ff', '#e6e6fa', '#dcdcdc'],
                'texture_complexity': 0.2,
                'lighting_characteristics': ['bright', 'reflected', 'uniform']
            },
            'tropical': {
                'description': 'Tropical jungle environments',
                'reference_images': [],
                'color_palette': ['#228b22', '#006400', '#8fbc8f', '#32cd32'],
                'texture_complexity': 0.9,
                'lighting_characteristics': ['intense', 'filtered', 'humid']
            }
        }
    
    def cache_processed_images(self, img: any, img_id: str, processing_params: Dict = None) -> str:
        """
        Cache processed images for faster subsequent access.
        
        Args:
            img: Image data to cache
            img_id: Unique identifier for the image
            processing_params: Processing parameters used
            
        Returns:
            Path to cached file
        """
        # Create cache key from image ID and processing parameters
        cache_key = self._generate_cache_key(img_id, processing_params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Save image data using pickle
            cache_data = {
                'image': img,
                'img_id': img_id,
                'processing_params': processing_params,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update cache database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            file_size = cache_file.stat().st_size
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache (cache_key, file_path, file_size)
                VALUES (?, ?, ?)
            ''', (cache_key, str(cache_file), file_size))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cached image: {img_id}")
            return str(cache_file)
            
        except Exception as e:
            logger.error(f"Failed to cache image {img_id}: {str(e)}")
            raise
    
    def load_cached_image(self, img_id: str, processing_params: Dict = None) -> Optional[any]:
        """
        Load cached processed image.
        
        Args:
            img_id: Image identifier
            processing_params: Processing parameters
            
        Returns:
            Cached image data or None if not found
        """
        cache_key = self._generate_cache_key(img_id, processing_params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Update access count
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE cache SET access_count = access_count + 1
                WHERE cache_key = ?
            ''', (cache_key,))
            conn.commit()
            conn.close()
            
            logger.debug(f"Loaded cached image: {img_id}")
            return cache_data['image']
            
        except Exception as e:
            logger.error(f"Failed to load cached image {img_id}: {str(e)}")
            return None
    
    def _generate_cache_key(self, img_id: str, processing_params: Dict = None) -> str:
        """Generate unique cache key"""
        key_data = f"{img_id}_{json.dumps(processing_params, sort_keys=True) if processing_params else ''}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def save_test_results(self, session_name: str, test_name: str, results: Dict[str, Any]) -> str:
        """
        Save final test results to database and file.
        
        Args:
            session_name: Name of test session
            test_name: Name of specific test
            results: Test results data
            
        Returns:
            Path to results file
        """
        try:
            # Find session ID
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM sessions WHERE session_name = ?', (session_name,))
            session_row = cursor.fetchone()
            
            if not session_row:
                raise ValueError(f"Session not found: {session_name}")
            
            session_id = session_row[0]
            
            # Save results to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"{test_name}_{timestamp}.json"
            results_file = self.results_dir / results_filename
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save to database
            cursor.execute('''
                INSERT INTO test_results (
                    session_id, test_name, image_path, overall_score,
                    color_score, pattern_score, brightness_score, distance_score,
                    environment_type, results_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, test_name, results.get('image_path'),
                results.get('overall_score'), results.get('color_score'),
                results.get('pattern_score'), results.get('brightness_score'),
                results.get('distance_score'), results.get('environment_type'),
                str(results_file)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved test results: {test_name}")
            return str(results_file)
            
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")
            raise
    
    def cleanup_old_cache(self, max_age_days: int = 30):
        """Clean up old cache files"""
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find old cache files
            cursor.execute('''
                SELECT file_path FROM cache 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(max_age_days))
            
            old_files = cursor.fetchall()
            
            for (file_path,) in old_files:
                cache_file = Path(file_path)
                if cache_file.exists():
                    cache_file.unlink()
            
            # Remove from database
            cursor.execute('''
                DELETE FROM cache WHERE created_at < datetime('now', '-{} days')
            '''.format(max_age_days))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {len(old_files)} old cache files")
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {str(e)}")
    
    def get_session_summary(self, session_name: str) -> Dict[str, Any]:
        """Get summary of a test session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute('''
                SELECT s.*, COUNT(tr.id) as test_count
                FROM sessions s
                LEFT JOIN test_results tr ON s.id = tr.session_id
                WHERE s.session_name = ?
                GROUP BY s.id
            ''', (session_name,))
            
            session_data = cursor.fetchone()
            
            if not session_data:
                return {}
            
            # Get test results
            cursor.execute('''
                SELECT test_name, overall_score, environment_type, timestamp
                FROM test_results 
                WHERE session_id = ?
                ORDER BY timestamp DESC
            ''', (session_data[0],))
            
            test_results = cursor.fetchall()
            
            conn.close()
            
            return {
                'session_name': session_data[1],
                'created_at': session_data[2],
                'description': session_data[3],
                'test_count': session_data[5],
                'tests': [
                    {
                        'name': test[0],
                        'score': test[1],
                        'environment': test[2],
                        'timestamp': test[3]
                    } for test in test_results
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {str(e)}")
            return {}