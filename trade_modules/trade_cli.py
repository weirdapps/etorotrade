#!/usr/bin/env python3
"""
CLI interface module for the trading system.
Extracted from trade.py to improve maintainability.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from yahoofinance.core.di_container import with_logger
from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.logging import get_logger
from yahoofinance.utils.dependency_injection import registry
from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.presentation import MarketDisplay

from .cli import get_user_source_choice
from .utils import get_file_paths


# Global configuration
INPUT_DIR = "input"
logger = get_logger(__name__)


class ConfigurationValidator:
    """Validates system configuration and environment variables."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_environment_variables(self) -> bool:
        """Validate required environment variables with proper sanitization."""
        required_vars = []  # Add required env vars here
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                self.errors.append(f"Missing required environment variable: {var}")
                continue
            
            # Validate environment variable value
            if not self._is_safe_env_value(value):
                self.errors.append(f"Environment variable {var} contains unsafe characters")
        
        return len(self.errors) == 0
    
    def _is_safe_env_value(self, value: str) -> bool:
        """Check if environment variable value is safe."""
        # Add validation logic for environment variables
        if not value or len(value) > 1000:  # Reasonable length limit
            return False
        
        # Check for obvious injection attempts
        dangerous_patterns = ['$(', '`', '&&', '||', ';', '|']
        return not any(pattern in value for pattern in dangerous_patterns)
    
    def validate_file_paths(self) -> bool:
        """Validate file paths using pathlib for security."""
        try:
            output_dir, input_dir, _, _, _ = get_file_paths()
            
            # Use pathlib for secure path handling
            output_path = Path(output_dir).resolve()
            input_path = Path(input_dir).resolve()
            
            # Ensure paths are within project directory
            project_root = Path.cwd().resolve()
            
            if not self._is_safe_path(output_path, project_root):
                self.errors.append(f"Output directory outside project: {output_path}")
            
            if not self._is_safe_path(input_path, project_root):
                self.errors.append(f"Input directory outside project: {input_path}")
                
        except Exception as e:
            self.errors.append(f"Path validation error: {str(e)}")
            return False
        
        return len(self.errors) == 0
    
    def _is_safe_path(self, path: Path, project_root: Path) -> bool:
        """Check if path is safe (within project boundaries)."""
        try:
            path.resolve().relative_to(project_root)
            return True
        except ValueError:
            return False
    
    def print_validation_report(self) -> bool:
        """Print validation report and return success status."""
        self.errors.clear()
        self.warnings.clear()
        
        self.validate_environment_variables()
        self.validate_file_paths()
        
        if self.errors:
            print("❌ Configuration Errors:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("⚠️  Configuration Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ Configuration validation passed")
        
        return len(self.errors) == 0


class ErrorSummaryCollector:
    """Collects and summarizes errors during execution."""
    
    def __init__(self):
        self.errors = []
    
    def add_error(self, error: str, context: str = ""):
        """Add error with context information."""
        error_entry = {"error": error, "context": context}
        self.errors.append(error_entry)
    
    def get_summary(self) -> str:
        """Get formatted error summary."""
        if not self.errors:
            return ""
        
        summary = "\n🔍 Error Summary:\n"
        for i, error_entry in enumerate(self.errors, 1):
            context_str = f" ({error_entry['context']})" if error_entry['context'] else ""
            summary += f"  {i}. {error_entry['error']}{context_str}\n"
        
        return summary


# Global instances
config_validator = ConfigurationValidator()
error_collector = ErrorSummaryCollector()


async def handle_trade_analysis(get_provider=None, app_logger=None):
    """Handle trade analysis workflow."""
    # Import the original function from backup
    from trade_original_backup import handle_trade_analysis as original_handle_trade_analysis
    
    # Call the original function with the same parameters
    return await original_handle_trade_analysis(get_provider=get_provider, app_logger=app_logger)


async def handle_portfolio_download(get_provider=None, app_logger=None):
    """
    Handle portfolio download based on user input.

    Args:
        get_provider: Injected provider factory function
        app_logger: Injected logger component

    Returns:
        bool: True if portfolio is available, False otherwise
    """
    # Import the original function from backup
    from trade_original_backup import handle_portfolio_download as original_handle_portfolio_download
    
    # Call the original function with the same parameters
    return await original_handle_portfolio_download(get_provider=get_provider, app_logger=app_logger)



async def main_async(get_provider=None, app_logger=None):
    """
    Async main function for CLI interface.
    
    Args:
        get_provider: Injected provider factory function
        app_logger: Injected logger component
    """
    display = None
    try:
        # Ensure we have the required dependencies
        provider = None
        if get_provider:
            if callable(get_provider):
                provider = get_provider(async_mode=True, max_concurrency=10)
                app_logger.info(f"Using injected provider from factory: {provider.__class__.__name__}")
            else:
                provider = get_provider
                app_logger.info(f"Using injected provider instance: {provider.__class__.__name__}")
        else:
            app_logger.error("Provider not injected, creating default provider")
            provider = AsyncHybridProvider(max_concurrency=10)

        # Override provider with AsyncHybridProvider for consistency
        if not isinstance(provider, AsyncHybridProvider):
            app_logger.info(f"Switching from {provider.__class__.__name__} to AsyncHybridProvider for consistency")
            provider = AsyncHybridProvider(max_concurrency=10)

        app_logger.info("Creating MarketDisplay instance...")
        display = MarketDisplay(provider=provider)
        app_logger.info("MarketDisplay created successfully")

        # Use the CLI module for user interaction
        source = get_user_source_choice()
        app_logger.info(f"User selected option: {source}")

        # Handle trade analysis separately
        if source == "T":
            app_logger.info("Handling trade analysis...")
            await handle_trade_analysis(get_provider=get_provider, app_logger=app_logger)
            return

        # Handle portfolio download if needed
        if source == "P":
            app_logger.info("Handling portfolio download...")
            if not await handle_portfolio_download(get_provider=get_provider, app_logger=app_logger):
                app_logger.error("Portfolio download failed")
                return
            app_logger.info("Portfolio download completed successfully")

        # Load tickers and display report
        app_logger.info(f"Loading tickers for source: {source}...")
        tickers = display.load_tickers(source)
        app_logger.info(f"Loaded {len(tickers)} tickers")

        app_logger.info("Displaying report...")
        # Pass verbose=True flag for eToro source and Manual Input due to special processing requirements
        verbose = source == "E" or source == "I"
        
        # Import the original display function
        from trade_original_backup import display_report_for_source
        
        # Use the proper display report function like the original
        await display_report_for_source(
            display,
            tickers,
            source,
            verbose=verbose,
            get_provider=get_provider,
            app_logger=app_logger,
        )

    except YFinanceError as e:
        error_collector.add_error(f"YFinance error in main_async: {str(e)}", context="main_async")
        if app_logger:
            app_logger.error(f"YFinance error in main_async: {str(e)}")
    except Exception as e:
        error_collector.add_error(f"Unexpected error in main_async: {str(e)}", context="main_async")
        if app_logger:
            app_logger.error(f"Unexpected error in main_async: {str(e)}")
    finally:
        # Cleanup
        if display:
            try:
                await display.close()
                if app_logger:
                    app_logger.info("Display cleanup completed successfully")
            except YFinanceError as e:
                if app_logger:
                    app_logger.debug(f"Error during cleanup: {str(e)}")
            except Exception as e:
                if app_logger:
                    app_logger.debug(f"Error during cleanup: {str(e)}")


@with_logger
def main(app_logger=None):
    """
    Command line interface entry point with dependency injection.
    
    Args:
        app_logger: Injected logger component
    """
    # Ensure output directories exist with secure permissions
    output_dir, input_dir, _, _, _ = get_file_paths()
    
    # Use pathlib for secure directory creation
    Path(output_dir).mkdir(parents=True, exist_ok=True, mode=0o755)
    Path(input_dir).mkdir(parents=True, exist_ok=True, mode=0o755)

    # Use inputs from v1 directory if available
    v1_input_dir = INPUT_DIR
    if os.path.exists(v1_input_dir):
        if app_logger:
            app_logger.debug(f"Using input files from legacy directory: {v1_input_dir}")
        else:
            logger.debug(f"Using input files from legacy directory: {v1_input_dir}")

    # Handle command line arguments if provided
    if len(sys.argv) > 1:
        # Basic argument handling for "trade.py i nvda" format
        source = sys.argv[1].upper()
        if source == "I" and len(sys.argv) > 2:
            tickers = sys.argv[2:]
            
            print(f"Processing manual input for tickers: {', '.join(tickers)}")

            # Get provider using dependency injection
            try:
                provider = registry.resolve("get_provider")(async_mode=True, max_concurrency=10)
                if app_logger:
                    app_logger.info(f"Using injected provider for manual input: {provider.__class__.__name__}")
            except Exception as e:
                if app_logger:
                    app_logger.error(f"Failed to resolve provider, using default: {str(e)}")
                # Fallback to direct instantiation
                provider = AsyncHybridProvider(max_concurrency=10)

            display = MarketDisplay(provider=provider)

            # Display report directly for manual tickers
            try:
                # Import the original display function
                from trade_original_backup import display_report_for_source
                
                # Use the proper display report function like the original
                asyncio.run(
                    display_report_for_source(
                        display,
                        tickers,
                        "I",
                        verbose=True,
                        get_provider=provider,
                        app_logger=app_logger,
                    )
                )
                return
            except Exception as e:
                error_collector.add_error(f"Error displaying individual report: {str(e)}", context="manual_tickers")
                if app_logger:
                    app_logger.error(f"Error displaying individual report: {str(e)}")
                return

    # Run async main for interactive mode
    try:
        asyncio.run(main_async(get_provider=None, app_logger=app_logger))
    except Exception as e:
        error_collector.add_error(f"Error in async main: {str(e)}", context="main_execution")
        if app_logger:
            app_logger.error(f"Error in async main: {str(e)}")


def setup_secure_file_copy():
    """Setup secure file copying from v1 to v2 directories."""
    output_dir, input_dir, _, _, _ = get_file_paths()
    
    # Ensure directories exist with secure permissions
    Path(output_dir).mkdir(parents=True, exist_ok=True, mode=0o755)
    Path(input_dir).mkdir(parents=True, exist_ok=True, mode=0o755)

    # Copy input files from v1 directory if they don't exist
    v1_input_path = Path(INPUT_DIR)
    input_path = Path(input_dir)
    
    if v1_input_path.exists():
        for file_path in v1_input_path.iterdir():
            if file_path.is_file():
                dst_file = input_path / file_path.name
                if not dst_file.exists():
                    # Secure file copy with proper permissions
                    import shutil
                    shutil.copy2(file_path, dst_file)
                    # Set secure permissions: owner read/write, group read, others no access
                    dst_file.chmod(0o640)
                    logger.debug(f"Copied {file_path.name} from v1 to v2 input directory with secure permissions")
    else:
        logger.debug(f"V1 input directory not found: {v1_input_path}")


if __name__ == "__main__":
    # Handle special validation-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-config":
        print("🔧 Running configuration validation...")
        is_valid = config_validator.print_validation_report()
        sys.exit(0 if is_valid else 1)
    
    try:
        # Run configuration validation first
        print("🔧 Running configuration validation...")
        if not config_validator.print_validation_report():
            print("\n❌ Configuration validation failed. Please fix the errors above before continuing.")
            print("💡 You can also run 'python trade_cli.py --validate-config' to check configuration without starting the application.")
            sys.exit(1)
        
        # Setup secure file operations
        setup_secure_file_copy()

        # Run the main function
        main()
    except YFinanceError as e:
        # Handle YFinance-specific errors
        error_collector.add_error(f"Critical error: {str(e)}", context="main_execution")
    except Exception as e:
        # Handle unexpected errors
        error_collector.add_error(f"Unexpected critical error: {str(e)}", context="main_execution")
    finally:
        # Always display error summary at the end
        summary = error_collector.get_summary()
        if summary:
            print(summary)