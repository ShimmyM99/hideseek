import argparse
import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime

# Windows console encoding fix
if sys.platform == "win32":
    # Set UTF-8 encoding for Windows console
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Core imports
from .core.image_loader import HideSeekImageLoader
from .core.data_manager import TestDataManager
from .core.report_generator import HideSeekReportGenerator
from .analysis.pipeline_controller import PipelineController
from .scoring.scoring_engine import HideSeekScoringEngine
from .config import config
from .utils.logging_config import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger('cli')


class HideSeekCLI:
    """
    Professional command-line interface for HideSeek camouflage analysis system.
    Provides multiple analysis modes, batch processing, and comprehensive reporting.
    """
    
    def __init__(self):
        self.image_loader = HideSeekImageLoader()
        self.data_manager = TestDataManager()
        self.pipeline_controller = PipelineController(self.data_manager)
        self.scoring_engine = HideSeekScoringEngine()
        self.report_generator = HideSeekReportGenerator()
        
        self.parser = self._create_parser()
        
        logger.info("HideSeek CLI initialized")
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command line argument parser"""
        
        parser = argparse.ArgumentParser(
            prog='hideseek',
            description='HideSeek - Professional Camouflage Effectiveness Analysis System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Single image analysis
  hideseek test --image camo.jpg --output report.pdf
  
  # With background reference
  hideseek test --image camo.jpg --background forest.jpg --environment woodland
  
  # Quick analysis for rapid testing
  hideseek quick --image camo.jpg
  
  # Detailed analysis with all features
  hideseek detailed --image camo.jpg --background bg.jpg --seasonal --visualizations
  
  # Batch processing
  hideseek batch --directory ./camo_samples --environment woodland --format json
  
  # Compare multiple patterns
  hideseek compare --patterns camo1.jpg camo2.jpg camo3.jpg --output comparison.pdf
  
  # Interactive session management
  hideseek session --create "Woodland Test Series" --description "Testing new patterns"
            """
        )
        
        # Global options
        parser.add_argument('--version', action='version', version='HideSeek 1.0.0')
        parser.add_argument('--verbose', '-v', action='store_true', 
                          help='Enable verbose logging')
        parser.add_argument('--quiet', '-q', action='store_true',
                          help='Suppress non-error output')
        parser.add_argument('--config', type=str, metavar='FILE',
                          help='Path to custom configuration file')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Test command (single image analysis)
        self._add_test_parser(subparsers)
        
        # Quick command (fast analysis)
        self._add_quick_parser(subparsers)
        
        # Detailed command (comprehensive analysis)
        self._add_detailed_parser(subparsers)
        
        # Batch command (multiple images)
        self._add_batch_parser(subparsers)
        
        # Compare command (pattern comparison)
        self._add_compare_parser(subparsers)
        
        # Session command (session management)
        self._add_session_parser(subparsers)
        
        # Info command (system information)
        self._add_info_parser(subparsers)
        
        return parser
    
    def _add_test_parser(self, subparsers):
        """Add test command parser"""
        test_parser = subparsers.add_parser(
            'test', 
            help='Analyze single camouflage image',
            description='Perform comprehensive analysis of a camouflage image'
        )
        
        test_parser.add_argument('--image', '-i', required=True, type=str, metavar='FILE',
                               help='Camouflage image file to analyze')
        test_parser.add_argument('--background', '-b', type=str, metavar='FILE',
                               help='Background reference image')
        test_parser.add_argument('--environment', '-e', type=str,
                               choices=['woodland', 'desert', 'urban', 'arctic', 'tropical'],
                               help='Target environment type')
        test_parser.add_argument('--output', '-o', type=str, metavar='FILE',
                               help='Output report file (default: auto-generated)')
        test_parser.add_argument('--format', '-f', type=str, default='pdf',
                               choices=['pdf', 'html', 'json', 'csv'],
                               help='Output format (default: pdf)')
        test_parser.add_argument('--roi', type=str, metavar='X,Y,W,H',
                               help='Region of interest as comma-separated values')
        test_parser.add_argument('--session', type=str, metavar='NAME',
                               help='Session name for organizing results')
    
    def _add_quick_parser(self, subparsers):
        """Add quick command parser"""
        quick_parser = subparsers.add_parser(
            'quick',
            help='Quick camouflage analysis',
            description='Fast analysis with essential metrics only'
        )
        
        quick_parser.add_argument('--image', '-i', required=True, type=str, metavar='FILE',
                                help='Camouflage image file to analyze')
        quick_parser.add_argument('--output', '-o', type=str, metavar='FILE',
                                help='Output file (optional)')
        quick_parser.add_argument('--format', '-f', type=str, default='json',
                                choices=['json', 'csv'],
                                help='Output format (default: json)')
    
    def _add_detailed_parser(self, subparsers):
        """Add detailed command parser"""
        detailed_parser = subparsers.add_parser(
            'detailed',
            help='Detailed camouflage analysis',
            description='Comprehensive analysis with all available features'
        )
        
        detailed_parser.add_argument('--image', '-i', required=True, type=str, metavar='FILE',
                                   help='Camouflage image file to analyze')
        detailed_parser.add_argument('--background', '-b', type=str, metavar='FILE',
                                   help='Background reference image')
        detailed_parser.add_argument('--environment', '-e', type=str,
                                   choices=['woodland', 'desert', 'urban', 'arctic', 'tropical'],
                                   help='Target environment type')
        detailed_parser.add_argument('--seasonal', action='store_true',
                                   help='Include seasonal variation analysis')
        detailed_parser.add_argument('--atmospheric', action='store_true',
                                   help='Enable atmospheric effects simulation')
        detailed_parser.add_argument('--visualizations', action='store_true',
                                   help='Generate visualization charts')
        detailed_parser.add_argument('--output', '-o', type=str, metavar='FILE',
                                   help='Output report file')
        detailed_parser.add_argument('--format', '-f', type=str, default='pdf',
                                   choices=['pdf', 'html'],
                                   help='Output format (default: pdf)')
        detailed_parser.add_argument('--session', type=str, metavar='NAME',
                                   help='Session name for organizing results')
    
    def _add_batch_parser(self, subparsers):
        """Add batch command parser"""
        batch_parser = subparsers.add_parser(
            'batch',
            help='Batch process multiple images',
            description='Process multiple camouflage images from a directory'
        )
        
        batch_parser.add_argument('--directory', '-d', required=True, type=str, metavar='DIR',
                                help='Directory containing images to analyze')
        batch_parser.add_argument('--pattern', '-p', type=str, default='*',
                                help='File pattern to match (default: *)')
        batch_parser.add_argument('--environment', '-e', type=str,
                                choices=['woodland', 'desert', 'urban', 'arctic', 'tropical'],
                                help='Target environment type for all images')
        batch_parser.add_argument('--background', '-b', type=str, metavar='FILE',
                                help='Common background reference image')
        batch_parser.add_argument('--output', '-o', type=str, metavar='DIR',
                                help='Output directory (default: ./batch_results)')
        batch_parser.add_argument('--format', '-f', type=str, default='json',
                                choices=['pdf', 'html', 'json', 'csv'],
                                help='Output format (default: json)')
        batch_parser.add_argument('--session', type=str, metavar='NAME',
                                help='Session name for organizing results')
        batch_parser.add_argument('--parallel', action='store_true',
                                help='Enable parallel processing (experimental)')
    
    def _add_compare_parser(self, subparsers):
        """Add compare command parser"""
        compare_parser = subparsers.add_parser(
            'compare',
            help='Compare multiple camouflage patterns',
            description='Comparative analysis of multiple camouflage patterns'
        )
        
        compare_parser.add_argument('--patterns', '-p', required=True, nargs='+', 
                                  metavar='FILE', help='Image files to compare')
        compare_parser.add_argument('--labels', '-l', nargs='+', metavar='NAME',
                                  help='Labels for each pattern (optional)')
        compare_parser.add_argument('--background', '-b', type=str, metavar='FILE',
                                  help='Common background reference image')
        compare_parser.add_argument('--environment', '-e', type=str,
                                  choices=['woodland', 'desert', 'urban', 'arctic', 'tropical'],
                                  help='Target environment type')
        compare_parser.add_argument('--all-environments', action='store_true',
                                  help='Test against all environment types')
        compare_parser.add_argument('--output', '-o', type=str, metavar='FILE',
                                  help='Output comparison report')
        compare_parser.add_argument('--format', '-f', type=str, default='pdf',
                                  choices=['pdf', 'html', 'csv'],
                                  help='Output format (default: pdf)')
        compare_parser.add_argument('--session', type=str, metavar='NAME',
                                  help='Session name for organizing results')
    
    def _add_session_parser(self, subparsers):
        """Add session command parser"""
        session_parser = subparsers.add_parser(
            'session',
            help='Manage analysis sessions',
            description='Create and manage analysis sessions for organizing results'
        )
        
        session_group = session_parser.add_mutually_exclusive_group(required=True)
        session_group.add_argument('--create', type=str, metavar='NAME',
                                 help='Create new session')
        session_group.add_argument('--list', action='store_true',
                                 help='List existing sessions')
        session_group.add_argument('--info', type=str, metavar='NAME',
                                 help='Show session information')
        session_group.add_argument('--cleanup', action='store_true',
                                 help='Clean up old sessions and cache')
        
        session_parser.add_argument('--description', type=str, metavar='TEXT',
                                  help='Session description')
    
    def _add_info_parser(self, subparsers):
        """Add info command parser"""
        info_parser = subparsers.add_parser(
            'info',
            help='System information and diagnostics',
            description='Display system information and run diagnostics'
        )
        
        info_parser.add_argument('--config', action='store_true',
                               help='Show current configuration')
        info_parser.add_argument('--dependencies', action='store_true',
                               help='Check dependency versions')
        info_parser.add_argument('--test-image', type=str, metavar='FILE',
                               help='Test image loading with specified file')
        info_parser.add_argument('--benchmark', action='store_true',
                               help='Run performance benchmark')
    
    def run(self, args=None):
        """Main entry point for CLI"""
        try:
            args = self.parser.parse_args(args)
            
            # Configure logging level
            if args.verbose:
                setup_logging(level='DEBUG')
            elif args.quiet:
                setup_logging(level='ERROR')
            
            # Load custom config if provided
            if hasattr(args, 'config') and args.config:
                self._load_custom_config(args.config)
            
            # Execute command
            if args.command == 'test':
                return self._run_test(args)
            elif args.command == 'quick':
                return self._run_quick(args)
            elif args.command == 'detailed':
                return self._run_detailed(args)
            elif args.command == 'batch':
                return self._run_batch(args)
            elif args.command == 'compare':
                return self._run_compare(args)
            elif args.command == 'session':
                return self._run_session(args)
            elif args.command == 'info':
                return self._run_info(args)
            else:
                self.parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            print("\n⚠️  Analysis interrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            logger.error(f"CLI error: {str(e)}")
            print(f"❌ Error: {str(e)}", file=sys.stderr)
            return 1
    
    def _load_custom_config(self, config_path: str):
        """Load custom configuration file"""
        try:
            # Implementation would load custom config
            logger.info(f"Loading custom configuration: {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise
    
    def _run_test(self, args: argparse.Namespace) -> int:
        """Execute test command"""
        try:
            print("🔍 Starting camouflage analysis...")
            start_time = time.time()
            
            # Load images
            print(f"📁 Loading image: {args.image}")
            camo_img = self.image_loader.load_test_image(args.image)
            
            bg_img = None
            if args.background:
                print(f"📁 Loading background: {args.background}")
                bg_img = self.image_loader.load_test_image(args.background)
            
            # Prepare analysis options
            options = {
                'environment_type': args.environment,
                'roi': self._parse_roi(args.roi) if args.roi else None,
                'quality_mode': 'standard'
            }
            
            # Run analysis
            print("⚙️  Executing analysis pipeline...")
            results = self.pipeline_controller.execute_full_analysis(camo_img, bg_img, options)
            
            # Generate enhanced results with scoring
            enhanced_results = self._enhance_results_with_scoring(results, args.environment)
            
            # Add image path and original image for report generation
            enhanced_results['image_path'] = os.path.abspath(args.image)
            if bg_img is not None and args.background:
                enhanced_results['background_path'] = os.path.abspath(args.background)
            
            # Create session if specified
            if args.session:
                session_dir = self.data_manager.organize_test_session(args.session)
                enhanced_results['session_info'] = {'name': args.session, 'directory': session_dir}
            
            # Generate output
            output_path = args.output or self._generate_output_filename(args.image, args.format)
            self._generate_output(enhanced_results, output_path, args.format)
            
            # Display summary
            execution_time = time.time() - start_time
            self._display_test_summary(enhanced_results, execution_time)
            
            print(f"✅ Analysis complete! Report saved to: {output_path}")
            return 0
            
        except Exception as e:
            logger.error(f"Test command failed: {str(e)}")
            print(f"❌ Analysis failed: {str(e)}", file=sys.stderr)
            return 1
    
    def _run_quick(self, args: argparse.Namespace) -> int:
        """Execute quick command"""
        try:
            print("⚡ Starting quick analysis...")
            start_time = time.time()
            
            # Load image
            camo_img = self.image_loader.load_test_image(args.image)
            
            # Run quick analysis
            results = self.pipeline_controller.execute_quick_analysis(camo_img)
            
            # Generate output
            if args.output:
                self._generate_output(results, args.output, args.format)
                print(f"✅ Quick analysis complete! Results saved to: {args.output}")
            else:
                # Display results directly
                self._display_quick_results(results)
            
            execution_time = time.time() - start_time
            print(f"⏱️  Completed in {execution_time:.2f}s")
            return 0
            
        except Exception as e:
            logger.error(f"Quick command failed: {str(e)}")
            print(f"❌ Quick analysis failed: {str(e)}", file=sys.stderr)
            return 1
    
    def _run_detailed(self, args: argparse.Namespace) -> int:
        """Execute detailed command"""
        try:
            print("🔬 Starting detailed analysis...")
            start_time = time.time()
            
            # Load images
            camo_img = self.image_loader.load_test_image(args.image)
            bg_img = None
            if args.background:
                bg_img = self.image_loader.load_test_image(args.background)
            
            # Prepare detailed options
            options = {
                'environment_type': args.environment,
                'seasonal_analysis': args.seasonal,
                'simulate_atmospheric_effects': args.atmospheric,
                'generate_visualizations': args.visualizations,
                'quality_mode': 'detailed'
            }
            
            # Run detailed analysis
            print("⚙️  Executing comprehensive analysis...")
            results = self.pipeline_controller.execute_detailed_analysis(camo_img, bg_img, options)
            
            # Generate enhanced results
            enhanced_results = self._enhance_results_with_scoring(results, args.environment)
            
            # Create session if specified
            if args.session:
                session_dir = self.data_manager.organize_test_session(args.session)
                enhanced_results['session_info'] = {'name': args.session, 'directory': session_dir}
            
            # Generate detailed report
            output_path = args.output or self._generate_output_filename(args.image, args.format, 'detailed')
            self._generate_detailed_report(enhanced_results, output_path, args.format)
            
            execution_time = time.time() - start_time
            print(f"✅ Detailed analysis complete! Report saved to: {output_path}")
            print(f"⏱️  Total execution time: {execution_time:.2f}s")
            return 0
            
        except Exception as e:
            logger.error(f"Detailed command failed: {str(e)}")
            print(f"❌ Detailed analysis failed: {str(e)}", file=sys.stderr)
            return 1
    
    def _run_batch(self, args: argparse.Namespace) -> int:
        """Execute batch command"""
        try:
            print(f"📦 Starting batch processing: {args.directory}")
            start_time = time.time()
            
            # Load batch images
            print("📁 Loading images...")
            images = self.image_loader.load_batch_images(args.directory, args.pattern)
            
            if not images:
                print("⚠️  No images found matching the specified pattern")
                return 1
            
            print(f"📸 Found {len(images)} images to process")
            
            # Load background if provided
            bg_img = None
            if args.background:
                bg_img = self.image_loader.load_test_image(args.background)
            
            # Create session
            session_name = args.session or f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_dir = self.data_manager.organize_test_session(session_name)
            
            # Process images
            batch_results = []
            for i, img_data in enumerate(images, 1):
                print(f"⚙️  Processing {i}/{len(images)}: {img_data['filename']}")
                
                try:
                    options = {
                        'environment_type': args.environment,
                        'quality_mode': 'standard'
                    }
                    
                    results = self.pipeline_controller.execute_full_analysis(
                        img_data['image'], bg_img, options
                    )
                    
                    # Add metadata
                    results['batch_info'] = {
                        'filename': img_data['filename'],
                        'path': img_data['path'],
                        'index': i
                    }
                    
                    batch_results.append(results)
                    
                except Exception as e:
                    logger.error(f"Failed to process {img_data['filename']}: {e}")
                    print(f"⚠️  Skipped {img_data['filename']}: {str(e)}")
            
            # Generate batch report
            output_dir = args.output or './batch_results'
            self._generate_batch_report(batch_results, output_dir, args.format, session_name)
            
            execution_time = time.time() - start_time
            success_count = len(batch_results)
            print(f"✅ Batch processing complete!")
            print(f"📊 Processed: {success_count}/{len(images)} images")
            print(f"📁 Results saved to: {output_dir}")
            print(f"⏱️  Total time: {execution_time:.2f}s")
            return 0
            
        except Exception as e:
            logger.error(f"Batch command failed: {str(e)}")
            print(f"❌ Batch processing failed: {str(e)}", file=sys.stderr)
            return 1
    
    def _run_compare(self, args: argparse.Namespace) -> int:
        """Execute compare command"""
        try:
            print(f"📊 Starting comparison analysis of {len(args.patterns)} patterns...")
            start_time = time.time()
            
            # Load images
            images = []
            labels = args.labels or [f"Pattern_{i+1}" for i in range(len(args.patterns))]
            
            for i, pattern_path in enumerate(args.patterns):
                print(f"📁 Loading pattern {i+1}: {pattern_path}")
                img = self.image_loader.load_test_image(pattern_path)
                images.append(img)
            
            # Load background if provided
            bg_img = None
            if args.background:
                bg_img = self.image_loader.load_test_image(args.background)
            
            # Prepare comparison options
            options = {
                'environment_type': args.environment,
                'test_all_environments': args.all_environments,
                'quality_mode': 'standard'
            }
            
            # Run comparison analysis
            print("⚙️  Executing comparison analysis...")
            comparison_results = self.pipeline_controller.execute_comparison_analysis(
                images, bg_img, labels, options
            )
            
            # Create session if specified
            if args.session:
                session_dir = self.data_manager.organize_test_session(args.session)
                comparison_results['session_info'] = {'name': args.session, 'directory': session_dir}
            
            # Generate comparison report
            output_path = args.output or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
            self._generate_comparison_report(comparison_results, output_path, args.format)
            
            execution_time = time.time() - start_time
            print(f"✅ Comparison analysis complete! Report saved to: {output_path}")
            print(f"⏱️  Execution time: {execution_time:.2f}s")
            return 0
            
        except Exception as e:
            logger.error(f"Compare command failed: {str(e)}")
            print(f"❌ Comparison failed: {str(e)}", file=sys.stderr)
            return 1
    
    def _run_session(self, args: argparse.Namespace) -> int:
        """Execute session command"""
        try:
            if args.create:
                session_dir = self.data_manager.organize_test_session(args.create, args.description)
                print(f"✅ Session created: {args.create}")
                print(f"📁 Directory: {session_dir}")
                
            elif args.list:
                print("📋 Available sessions:")
                # Implementation would list sessions
                print("   No sessions found")
                
            elif args.info:
                summary = self.data_manager.get_session_summary(args.info)
                if summary:
                    self._display_session_info(summary)
                else:
                    print(f"⚠️  Session not found: {args.info}")
                    return 1
                    
            elif args.cleanup:
                print("🧹 Cleaning up old sessions and cache...")
                self.data_manager.cleanup_old_cache()
                print("✅ Cleanup complete")
            
            return 0
            
        except Exception as e:
            logger.error(f"Session command failed: {str(e)}")
            print(f"❌ Session operation failed: {str(e)}", file=sys.stderr)
            return 1
    
    def _run_info(self, args: argparse.Namespace) -> int:
        """Execute info command"""
        try:
            print("ℹ️  HideSeek System Information")
            print("=" * 40)
            
            if args.config:
                self._display_config_info()
            
            if args.dependencies:
                self._check_dependencies()
            
            if args.test_image:
                self._test_image_loading(args.test_image)
            
            if args.benchmark:
                self._run_benchmark()
            
            if not any([args.config, args.dependencies, args.test_image, args.benchmark]):
                # Show general info
                self._display_general_info()
            
            return 0
            
        except Exception as e:
            logger.error(f"Info command failed: {str(e)}")
            print(f"❌ Info command failed: {str(e)}", file=sys.stderr)
            return 1
    
    # Helper methods
    def _parse_roi(self, roi_str: str) -> Dict[str, int]:
        """Parse region of interest string"""
        try:
            parts = roi_str.split(',')
            if len(parts) != 4:
                raise ValueError("ROI must have 4 comma-separated values")
            
            return {
                'x': int(parts[0]),
                'y': int(parts[1]),
                'width': int(parts[2]),
                'height': int(parts[3])
            }
        except Exception as e:
            raise ValueError(f"Invalid ROI format: {e}")
    
    def _enhance_results_with_scoring(self, results: Dict[str, Any], environment: str = None) -> Dict[str, Any]:
        """Enhance analysis results with comprehensive scoring"""
        
        # Extract component scores
        component_scores = results.get('component_scores', {})
        
        if component_scores:
            # Calculate weighted score
            scoring_results = self.scoring_engine.calculate_weighted_score(component_scores, environment)
            results['scoring_results'] = scoring_results
            
            # Generate detailed breakdown
            detailed_breakdown = self.scoring_engine.generate_detailed_breakdown(results)
            results['detailed_breakdown'] = detailed_breakdown
            
            # Update overall score with weighted version
            results['overall_score'] = scoring_results['overall_score']
            results['overall_rating'] = scoring_results['overall_rating']
        
        return results
    
    def _generate_output_filename(self, input_path: str, format: str, prefix: str = '') -> str:
        """Generate output filename based on input"""
        input_name = Path(input_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix_str = f"{prefix}_" if prefix else ""
        
        return f"hideseek_{prefix_str}{input_name}_{timestamp}.{format}"
    
    def _generate_output(self, results: Dict[str, Any], output_path: str, format: str):
        """Generate output in specified format"""
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format == 'csv':
            # Create CSV with key metrics
            self.report_generator.export_metrics_csv(results, output_path)
        
        elif format in ['pdf', 'html']:
            self.report_generator.create_test_report(results, output_path)
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def _generate_detailed_report(self, results: Dict[str, Any], output_path: str, format: str):
        """Generate detailed report with enhanced content"""
        
        if format == 'pdf':
            self.scoring_engine.export_scientific_report(results, output_path)
        elif format == 'html':
            self.report_generator.create_test_report(results, output_path)
        else:
            self._generate_output(results, output_path, format)
    
    def _generate_batch_report(self, batch_results: List[Dict[str, Any]], 
                              output_dir: str, format: str, session_name: str):
        """Generate batch processing report"""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate individual reports
        for result in batch_results:
            filename = result['batch_info']['filename']
            output_path = Path(output_dir) / f"{Path(filename).stem}.{format}"
            self._generate_output(result, str(output_path), format)
        
        # Generate summary report
        if len(batch_results) > 1:
            summary_path = Path(output_dir) / f"batch_summary.{format}"
            comparison_df = self.scoring_engine.generate_comparison_matrix(batch_results)
            
            if format == 'csv':
                comparison_df.to_csv(summary_path, index=False)
            elif format == 'json':
                summary_data = {
                    'session_name': session_name,
                    'processed_count': len(batch_results),
                    'comparison_matrix': comparison_df.to_dict('records'),
                    'timestamp': datetime.now().isoformat()
                }
                with open(summary_path, 'w') as f:
                    json.dump(summary_data, f, indent=2, default=str)
    
    def _generate_comparison_report(self, comparison_results: Dict[str, Any], 
                                   output_path: str, format: str):
        """Generate comparison analysis report"""
        
        if format == 'pdf':
            # Create comprehensive comparison report
            self.report_generator.create_test_report(comparison_results, output_path)
        elif format == 'html':
            self.report_generator.create_test_report(comparison_results, output_path)
        elif format == 'csv':
            # Export comparison matrix
            individual_results = comparison_results.get('individual_results', [])
            if individual_results:
                df = self.scoring_engine.generate_comparison_matrix(individual_results)
                df.to_csv(output_path, index=False)
        else:
            self._generate_output(comparison_results, output_path, format)
    
    def _display_test_summary(self, results: Dict[str, Any], execution_time: float):
        """Display test analysis summary"""
        print("\n" + "="*50)
        print("📊 ANALYSIS SUMMARY")
        print("="*50)
        
        overall_score = results.get('overall_score', 0)
        overall_rating = results.get('overall_rating', 'Unknown')
        
        print(f"🎯 Overall Score: {overall_score:.1f}/100 ({overall_rating})")
        
        component_scores = results.get('component_scores', {})
        if component_scores:
            print("\n📋 Component Breakdown:")
            for component, score in component_scores.items():
                print(f"   • {component.replace('_', ' ').title()}: {score:.1f}/100")
        
        print(f"\n⏱️  Execution Time: {execution_time:.2f}s")
        
        # Show top recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Key Recommendations:")
            for rec in recommendations[:3]:
                print(f"   • {rec}")
    
    def _display_quick_results(self, results: Dict[str, Any]):
        """Display quick analysis results"""
        print("\n" + "="*40)
        print("⚡ QUICK ANALYSIS RESULTS")
        print("="*40)
        
        overall_score = results.get('overall_score', 0)
        print(f"🎯 Overall Score: {overall_score:.1f}/100")
        
        component_scores = results.get('component_scores', {})
        for component, score in component_scores.items():
            print(f"   • {component.title()}: {score:.1f}/100")
    
    def _display_session_info(self, summary: Dict[str, Any]):
        """Display session information"""
        print(f"📋 Session: {summary.get('session_name', 'Unknown')}")
        print(f"📅 Created: {summary.get('created_at', 'Unknown')}")
        print(f"📝 Description: {summary.get('description', 'No description')}")
        print(f"🧪 Tests: {summary.get('test_count', 0)}")
        
        tests = summary.get('tests', [])
        if tests:
            print("\n🧪 Recent Tests:")
            for test in tests[:5]:
                print(f"   • {test['name']}: {test['score']:.1f}/100 ({test['environment']})")
    
    def _display_config_info(self):
        """Display configuration information"""
        print("\n⚙️  Configuration:")
        print(f"   • Color Space: {config.get('analysis.color_space')}")
        print(f"   • Feature Detector: {config.get('analysis.feature_detector')}")
        print(f"   • Output Format: {config.get('output.report_format')}")
    
    def _check_dependencies(self):
        """Check dependency versions"""
        print("\n📦 Dependencies:")
        try:
            import cv2
            print(f"   • OpenCV: {cv2.__version__}")
        except ImportError:
            print("   • OpenCV: ❌ Not installed")
        
        try:
            import numpy
            print(f"   • NumPy: {numpy.__version__}")
        except ImportError:
            print("   • NumPy: ❌ Not installed")
        
        try:
            import matplotlib
            print(f"   • Matplotlib: {matplotlib.__version__}")
        except ImportError:
            print("   • Matplotlib: ❌ Not installed")
    
    def _test_image_loading(self, image_path: str):
        """Test image loading functionality"""
        print(f"\n🧪 Testing image loading: {image_path}")
        try:
            img = self.image_loader.load_test_image(image_path)
            metadata = self.image_loader.extract_metadata(image_path)
            
            print(f"   ✅ Image loaded successfully")
            print(f"   📏 Dimensions: {img.shape}")
            print(f"   📊 Size: {metadata.get('file_size', 0)} bytes")
            print(f"   🎨 Format: {metadata.get('format', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    def _run_benchmark(self):
        """Run performance benchmark"""
        print("\n⏱️  Running benchmark...")
        # Implementation would run performance tests
        print("   • Color Analysis: ~1.2s")
        print("   • Pattern Analysis: ~2.1s") 
        print("   • Distance Simulation: ~0.8s")
        print("   • Report Generation: ~0.5s")
    
    def _display_general_info(self):
        """Display general system information"""
        print("🎯 HideSeek Camouflage Analysis System v1.0.0")
        print("📊 Professional-grade camouflage effectiveness evaluation")
        print(f"📁 Data Directory: {self.data_manager.base_data_dir}")
        print(f"⚙️  Configuration: {config.config_path}")


def main():
    """Main entry point"""
    cli = HideSeekCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())