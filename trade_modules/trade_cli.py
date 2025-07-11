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
        
        # Silent validation - only return status
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


def get_provider_instance():
    """Get default provider instance for CLI operations."""
    try:
        # Try to get provider from dependency injection first
        provider = registry.resolve("get_provider")(async_mode=True, max_concurrency=10)
        return provider
    except Exception:
        # Fallback to direct instantiation
        return AsyncHybridProvider(max_concurrency=10)


async def handle_trade_analysis(get_provider=None, app_logger=None):
    """Handle trade analysis workflow."""
    try:
        # Get provider instance
        if get_provider:
            if callable(get_provider):
                provider = get_provider()
            else:
                provider = get_provider
        else:
            provider = get_provider_instance()
        
        # Run market analysis using trade modules
        from trade import run_market_analysis
        from trade_modules.utils import get_file_paths
        import pandas as pd
        
        # Load market data
        paths = get_file_paths()
        market_df = pd.read_csv(paths[2])  # market.csv path
        
        # Run analysis
        opportunities = run_market_analysis(market_df, provider)
        
        if app_logger:
            app_logger.info("Trade analysis completed successfully")
        
        return opportunities
        
    except Exception as e:
        if app_logger:
            app_logger.error(f"Trade analysis failed: {str(e)}")
        raise


async def handle_trade_analysis_direct(display, trade_choice, get_provider=None, app_logger=None):
    """
    Handle trade analysis by using existing data from CSV files (no API calls).
    
    Args:
        display: MarketDisplay instance
        trade_choice: Trade choice (B, S, or H)
        get_provider: Provider factory function (optional)
        app_logger: Logger instance (optional)
    """
    import pandas as pd
    import os
    from trade_modules.utils import get_file_paths
    
    try:
        # Get file paths
        output_dir, input_dir, market_file, portfolio_file, _ = get_file_paths()
        
        # Determine which file to load and filter logic
        if trade_choice == "B":
            # BUY: Check market.csv for buy opportunities NOT in portfolio or sell files
            data_file = market_file
            exclusion_files = [portfolio_file, os.path.join(output_dir, "sell.csv")]
            title = "Trade Analysis - BUY Opportunities (Market data excluding portfolio/notrade)"
            output_filename = "buy.csv"
            app_logger.info("Loading market data for BUY opportunities analysis")
            
        elif trade_choice == "S":
            # SELL: Check portfolio.csv for sell opportunities
            data_file = portfolio_file
            exclusion_files = []
            title = "Trade Analysis - SELL Opportunities (Portfolio data)"
            output_filename = "sell.csv"
            app_logger.info("Loading portfolio data for SELL opportunities analysis")
            
        elif trade_choice == "H":
            # HOLD: Check market.csv for hold opportunities NOT in portfolio or sell files
            data_file = market_file
            exclusion_files = [portfolio_file, os.path.join(output_dir, "sell.csv")]
            title = "Trade Analysis - HOLD Opportunities (Market data excluding portfolio/notrade)"
            output_filename = "hold.csv"
            app_logger.info("Loading market data for HOLD opportunities analysis")
            
        else:
            app_logger.error(f"Invalid trade choice: {trade_choice}")
            return
        
        # Check if data file exists
        if not os.path.exists(data_file):
            app_logger.error(f"Data file not found: {data_file}")
            return
        
        # Load and process data directly from CSV without API calls
        await display_existing_csv_data(data_file, exclusion_files, title, output_filename, trade_choice, app_logger)
        
        app_logger.info(f"Trade analysis {trade_choice} completed successfully")
        
    except Exception as e:
        app_logger.error(f"Trade analysis direct failed: {str(e)}")
        raise


async def display_existing_csv_data(data_file, exclusion_files, title, output_filename, trade_choice, app_logger):
    """
    Display existing CSV data with filtering and trade action analysis (no API calls).
    
    Args:
        data_file: Path to main data file
        exclusion_files: List of files containing tickers to exclude
        title: Display title
        output_filename: Output file name
        trade_choice: Trade choice (B, S, H)
        app_logger: Logger instance
    """
    import pandas as pd
    import os
    from trade_modules.utils import get_file_paths
    
    try:
        # Load the main data file
        df = pd.read_csv(data_file)
        app_logger.info(f"Loaded {len(df)} records from {data_file}")
        
        if df.empty:
            return
        
        # Load exclusion tickers if needed
        exclusion_tickers = set()
        if exclusion_files:
            for exclusion_file in exclusion_files:
                if os.path.exists(exclusion_file):
                    try:
                        excl_df = pd.read_csv(exclusion_file)
                        if 'symbol' in excl_df.columns:
                            exclusion_tickers.update(excl_df['symbol'].astype(str).str.upper())
                        elif 'TICKER' in excl_df.columns:
                            exclusion_tickers.update(excl_df['TICKER'].astype(str).str.upper())
                        app_logger.info(f"Loaded {len(excl_df)} exclusion tickers from {exclusion_file}")
                    except Exception as e:
                        pass  # Silent error handling
        
        # Apply exclusion filtering only for BUY and HOLD (not for SELL)
        if exclusion_tickers and trade_choice in ["B", "H"]:
            original_count = len(df)
            ticker_col = 'symbol' if 'symbol' in df.columns else 'TICKER'
            if ticker_col in df.columns:
                df = df[~df[ticker_col].astype(str).str.upper().isin(exclusion_tickers)]
                app_logger.info(f"After exclusion filtering: {len(df)} records (excluded {original_count - len(df)})")
        elif trade_choice == "S":
            app_logger.info(f"Processing portfolio data for SELL opportunities: {len(df)} records")
        
        # Apply trade action filtering to existing data
        if not df.empty:
            try:
                # Calculate actions for all rows
                from yahoofinance.utils.trade_criteria import calculate_action_for_row
                actions = []
                filtered_rows = []
                
                for _, row in df.iterrows():
                    try:
                        action, _ = calculate_action_for_row(row.to_dict(), {}, "short_percent")
                        if action == trade_choice:
                            actions.append(action)
                            filtered_rows.append(row.to_dict())
                    except Exception as e:
                        # If action calculation fails, include in HOLD filter only
                        if trade_choice == "H":
                            actions.append("H")
                            filtered_rows.append(row.to_dict())
                
                if filtered_rows:
                    # Use MarketDisplay to show the filtered data
                    from yahoofinance.presentation.console import MarketDisplay
                    from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
                    
                    # Create a minimal provider (won't be used for API calls)
                    provider = AsyncHybridProvider(max_concurrency=1)
                    display = MarketDisplay(provider=provider)
                    
                    # Display the results
                    display.display_stock_table(filtered_rows, title)
                    
                    # Display processing statistics after the table
                    from yahoofinance.utils.async_utils.enhanced import display_processing_stats
                    display_processing_stats()
                    
                    # Save to CSV
                    output_dir, _, _, _, _ = get_file_paths()
                    output_path = os.path.join(output_dir, output_filename)
                    display.save_to_csv(filtered_rows, output_filename)
                    
                    app_logger.info(f"Displayed {len(filtered_rows)} {trade_choice} opportunities")
                else:
                    app_logger.info(f"No {trade_choice} opportunities found after filtering")
                    
            except Exception as e:
                app_logger.error(f"Error filtering trade actions: {e}")
                # Fallback: display all data without action filtering
                
                # Convert DataFrame to list of dicts for display
                all_data = df.to_dict('records')
                from yahoofinance.presentation.console import MarketDisplay
                from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
                
                provider = AsyncHybridProvider(max_concurrency=1)
                display = MarketDisplay(provider=provider)
                display.display_stock_table(all_data, title)
                
                # Display processing statistics after the table
                from yahoofinance.utils.async_utils.enhanced import display_processing_stats
                display_processing_stats()
                
                output_dir, _, _, _, _ = get_file_paths()
                output_path = os.path.join(output_dir, output_filename)
                display.save_to_csv(all_data, output_filename)
        else:
            app_logger.info("No data remaining after exclusion filtering")
            
    except Exception as e:
        app_logger.error(f"Error displaying CSV data: {e}")
        raise


async def handle_portfolio_download(get_provider=None, app_logger=None):
    """
    Handle portfolio download based on user input.

    Args:
        get_provider: Injected provider factory function
        app_logger: Injected logger component

    Returns:
        bool: True if portfolio is available, False otherwise
    """
    try:
        # Import download functions
        from yahoofinance.data.download import download_portfolio
        
        # Get provider instance
        if get_provider:
            if callable(get_provider):
                provider = get_provider()
            else:
                provider = get_provider
        else:
            provider = get_provider_instance()
        
        # Download portfolio data
        success = await download_portfolio(provider)
        
        if app_logger:
            if success:
                app_logger.info("Portfolio download completed successfully")
            else:
                app_logger.warning("Portfolio download failed or incomplete")
        
        return bool(success)
        
    except Exception as e:
        if app_logger:
            app_logger.error(f"Portfolio download failed: {str(e)}")
        return False


async def display_market_report(display, tickers, source, verbose=False, get_provider=None, app_logger=None, trade_filter=None):
    """
    Display market data report using MarketDisplay instance.
    
    Args:
        display: MarketDisplay instance
        tickers: List of ticker symbols
        source: Data source identifier
        verbose: Whether to display verbose output
        get_provider: Provider factory function (optional)
        app_logger: Logger instance (optional)
        trade_filter: Trade analysis filter (B, S, H) for filtering results
    """
    try:
        if app_logger:
            app_logger.info(f"Displaying market report for {len(tickers)} tickers from source {source}")
        
        # Use the MarketDisplay instance to show the report
        # Since we're in an async context, call the async method directly
        await display._async_display_report(tickers, report_type=source, trade_filter=trade_filter)
        
        if app_logger:
            app_logger.info("Market report displayed successfully")
            
    except Exception as e:
        if app_logger:
            app_logger.error(f"Failed to display market report: {str(e)}")
        raise



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

        # Handle portfolio flow with suboptions
        portfolio_choice = None
        if source == "P":
            app_logger.info("Handling portfolio flow...")
            # Import CLI functions for portfolio suboptions
            from .cli import get_portfolio_choice
            portfolio_choice = get_portfolio_choice()
            app_logger.info(f"Portfolio choice: {portfolio_choice}")
            
            # Only download new portfolio if user chose N
            if portfolio_choice == "N":
                app_logger.info("Downloading new portfolio...")
                if not await handle_portfolio_download(get_provider=get_provider, app_logger=app_logger):
                    app_logger.error("Portfolio download failed")
                    return
                app_logger.info("New portfolio download completed successfully")
            else:
                app_logger.info("Using existing portfolio file")

        # Handle trade analysis flow with suboptions
        trade_choice = None
        if source == "T":
            app_logger.info("Handling trade analysis flow...")
            # Import CLI functions for trade analysis suboptions
            from .cli import get_trade_analysis_choice
            trade_choice = get_trade_analysis_choice()
            app_logger.info(f"Trade analysis choice: {trade_choice}")
            
            # Load and display data directly from appropriate files
            await handle_trade_analysis_direct(display, trade_choice, get_provider, app_logger)
            return

        # Load tickers and display report
        app_logger.info(f"Loading tickers for source: {source}...")
        tickers = display.load_tickers(source)
        app_logger.info(f"Loaded {len(tickers)} tickers")

        app_logger.info("Displaying report...")
        # Pass verbose=True flag for eToro source and Manual Input due to special processing requirements
        verbose = source == "E" or source == "I"
        
        # Display the market data report
        await display_market_report(
            display,
            tickers,
            source,
            verbose=verbose,
            get_provider=get_provider,
            app_logger=app_logger,
            trade_filter=None,  # No trade filter for regular flow
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
                # Display the market data report for manual input
                asyncio.run(
                    display_market_report(
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
        is_valid = config_validator.print_validation_report()
        sys.exit(0 if is_valid else 1)
    
    try:
        # Run configuration validation first
        if not config_validator.print_validation_report():
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
        # Silent error collection - errors handled through logging
        pass