"""
Data downloading utilities.

This module provides functions for downloading data from various sources.
"""

import asyncio
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import aiofiles
import aiohttp
import pandas as pd
import requests
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..core.config import FILE_PATHS, PATHS
from ..core.logging import get_logger


logger = get_logger(__name__)


def fix_hk_ticker(ticker):
    """
    Fix HK stock tickers to standardize their format:
    1. If fewer than 4 numerals, add leading zeros to make it 4 numerals
    2. If more than 4 numerals and they are leading zeros, remove until you get to 4 numerals
    3. If more than 4 numerals and the leading numeral is not zero, keep as is

    Args:
        ticker: The ticker string to process

    Returns:
        The processed ticker with standardized format
    """
    # Return early if not a valid HK ticker format
    if not isinstance(ticker, str) or not ticker.endswith(".HK"):
        return ticker

    parts = ticker.split(".")
    if len(parts) != 2:
        return ticker

    numeric_part = parts[0]
    fixed_ticker = ticker

    # Process based on digit count
    if len(numeric_part) < 4:
        # Add leading zeros for fewer than 4 digits
        fixed_ticker = numeric_part.zfill(4) + ".HK"
    elif len(numeric_part) > 4 and numeric_part.startswith("0"):
        # Remove leading zeros for more than 4 digits
        stripped_part = numeric_part.lstrip("0")
        fixed_ticker = (stripped_part.zfill(4) if len(stripped_part) < 4 else stripped_part) + ".HK"

    # Log changes if the ticker was modified
    if fixed_ticker != ticker:
        logger.info(f"Fixed HK ticker: {ticker} -> {fixed_ticker}")

    return fixed_ticker


# Load environment variables
load_dotenv()


def safe_click(driver, element, description="element"):
    """Helper function to safely click an element using JavaScript"""
    try:
        time.sleep(1)  # Brief pause before clicking
        driver.execute_script("arguments[0].click();", element)
        logger.info(f"Clicked {description}")
        return True
    except ElementClickInterceptedException as e:
        logger.error(f"Click intercepted for {description}: {str(e)}")
        raise e
    except WebDriverException as e:
        logger.error(f"WebDriver error clicking {description}: {str(e)}")
        raise e


def setup_driver():
    """Setup Chrome WebDriver with secure options"""
    options = webdriver.ChromeOptions()
    # Security: Removed --no-sandbox and --disable-web-security
    # Only use these if absolutely necessary and in controlled environments
    options.add_argument("--disable-dev-shm-usage")
    # Add additional options for stability while maintaining security
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1200,800")
    # Add security-focused options
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-images")  # Faster loading
    options.add_argument("--disable-javascript")  # More secure if JS not needed
    # Set a specific user data directory within project (not system temp)
    import os
    user_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chrome_profile")
    os.makedirs(user_data_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={user_data_dir}")
    return webdriver.Chrome(options=options)


def wait_and_find_element(driver, by, value, timeout=10, check_visibility=True):
    """Helper function to wait for and find an element"""
    try:
        if check_visibility:
            element = WebDriverWait(driver, timeout).until(
                EC.visibility_of_element_located((by, value))
            )
        else:
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
        return element
    except (YFinanceError, TimeoutException, NoSuchElementException) as e:
        logger.error(f"Error finding element {value}: {str(e)}")
        return None


def find_sign_in_button(driver):
    """Helper function to find and click the sign-in button"""
    try:
        logger.info("Looking for sign-in button...")
        sign_in_elements = driver.find_elements(By.CLASS_NAME, "action-button")
        for element in sign_in_elements:
            if "Sign in" in element.text:
                logger.info(f"Found sign-in button with text: {element.text}")
                return element
        return None
    except NoSuchElementException as e:
        logger.error(f"Sign-in button not found: {str(e)}")
        raise e
    except WebDriverException as e:
        logger.error(f"WebDriver error finding sign-in button: {str(e)}")
        raise e


def handle_email_sign_in(driver):
    """Handle the email sign-in process"""
    logger.info("Looking for email sign-in button...")
    email_sign_in = wait_and_find_element(
        driver, By.XPATH, "//button[contains(., 'Sign in with email')]"
    )
    if not email_sign_in:
        raise NoSuchElementException("Email sign-in button not found")
    safe_click(driver, email_sign_in, "email sign-in button")
    time.sleep(5)


def handle_email_input(driver, email):
    """Handle the email input and next button process"""
    logger.info("Looking for email input...")
    email_input = wait_and_find_element(driver, By.ID, "ui-sign-in-email-input")
    if not email_input:
        email_input = wait_and_find_element(driver, By.XPATH, "//input[@type='email']")
    if not email_input:
        raise NoSuchElementException("Email input field not found")

    logger.info("Entering email...")
    email_input.clear()
    time.sleep(1)
    email_input.send_keys(email)
    time.sleep(2)

    # Look for the Next button
    logger.info("Looking for Next button...")
    next_button = wait_and_find_element(
        driver, By.XPATH, "//button[contains(text(), 'Next') or contains(text(), 'NEXT')]"
    )
    if next_button:
        safe_click(driver, next_button, "Next button")
    else:
        logger.info("No Next button found, submitting form...")
        email_input.send_keys(Keys.RETURN)
        logger.info("Sent RETURN key to email input")

    time.sleep(5)  # Wait longer for password field


def find_password_input(driver, timeout=5):
    """Try different selectors to find password input"""
    for selector in [
        (By.XPATH, "//input[@type='password']"),
        (By.CSS_SELECTOR, "input[type='password']"),
        (By.XPATH, "//input[contains(@class, 'password')]"),
    ]:
        try:
            password_input = wait_and_find_element(
                driver, selector[0], selector[1], timeout=timeout
            )
            if password_input:
                return password_input
        except (NoSuchElementException, TimeoutException):
            continue
    return None


def handle_password_submit(driver, password_input, password):
    """Handle password submission and final sign in"""
    logger.info("Found password input, entering password...")

    # Clear any existing text in the password field
    time.sleep(1)
    password_input.clear()

    # Enter password
    password_input.send_keys(password)
    logger.info(f"Entered password (length: {len(password)})")
    time.sleep(2)

    # Try multiple different selectors for the sign-in button
    logger.info("Looking for sign-in button after password entry...")
    sign_in_button = None

    # List of different button selectors to try
    button_selectors = [
        # Exact match for the specific button class from pi-screener
        (By.CSS_SELECTOR, "button.firebaseui-id-submit"),
        (By.CSS_SELECTOR, "button.mdl-button--colored[type='submit']"),
        (By.CSS_SELECTOR, "button.firebaseui-button"),
        (By.CSS_SELECTOR, "button.mdl-button--raised"),
        # More general selectors as fallbacks
        (By.XPATH, "//button[@type='submit']"),
        (By.XPATH, "//button[contains(text(), 'Sign in')]"),
        (By.XPATH, "//button[contains(text(), 'SIGN IN')]"),
        (By.XPATH, "//button[contains(text(), 'Sign In')]"),
        (By.XPATH, "//button[contains(text(), 'Login')]"),
        (By.XPATH, "//button[contains(text(), 'Log in')]"),
        (By.XPATH, "//button[contains(@class, 'signin')]"),
        (By.XPATH, "//button[contains(@class, 'login')]"),
        (By.XPATH, "//input[@type='submit']"),
        (By.CSS_SELECTOR, "button.sign-in"),
        (By.CSS_SELECTOR, "button.login"),
        (By.CSS_SELECTOR, "input[type='submit']"),
        # Try finding buttons with role='button'
        (By.XPATH, "//div[@role='button' and contains(text(), 'Sign in')]"),
        (By.XPATH, "//div[@role='button' and contains(text(), 'Login')]"),
    ]

    # Try each selector
    for selector_type, selector in button_selectors:
        try:
            button = driver.find_element(selector_type, selector)
            if button:
                sign_in_button = button
                logger.info(f"Found sign-in button with selector: {selector}")
                break
        except NoSuchElementException:
            continue

    # Click the sign-in button if found
    if sign_in_button:
        logger.info("Attempting to click sign-in button")
        try:
            # First try with safe_click
            safe_click(driver, sign_in_button, "Sign in button")
        except Exception as e:
            logger.warning(f"Error with safe_click: {str(e)}, trying direct click")
            try:
                # Then try direct click if JavaScript click fails
                sign_in_button.click()
                logger.info("Direct click succeeded")
            except Exception as e2:
                logger.warning(f"Direct click failed: {str(e2)}, trying Enter key")
                password_input.send_keys(Keys.RETURN)
    else:
        # Try looking for a form instead and submit it
        logger.info("No sign-in button found, looking for form to submit...")
        try:
            form = driver.find_element(By.TAG_NAME, "form")
            if form:
                logger.info("Found form, submitting with JavaScript")
                try:
                    driver.execute_script("arguments[0].submit();", form)
                    logger.info("Form submitted successfully")
                except Exception as e:
                    logger.warning(f"Error submitting form with JavaScript: {str(e)}")
                    logger.info("Trying to submit with Enter key instead")
                    password_input.send_keys(Keys.RETURN)
            else:
                logger.info("No form found, submitting with Enter key")
                password_input.send_keys(Keys.RETURN)
        except NoSuchElementException:
            logger.info("No form found, submitting with Enter key")
            password_input.send_keys(Keys.RETURN)

    logger.info("Waiting for login to complete...")
    time.sleep(15)  # Wait longer for login to complete


def handle_password_input(driver, password, max_attempts=3):
    """Handle the password input and final sign in process"""
    logger.info("Looking for password input...")
    last_error = None

    for attempt in range(max_attempts):
        try:
            password_input = find_password_input(driver)
            if password_input:
                handle_password_submit(driver, password_input, password)
                return True
        except (
            NoSuchElementException,
            TimeoutException,
            ElementClickInterceptedException,
            WebDriverException,
        ) as e:
            last_error = e
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_attempts - 1:
                logger.info("Retrying...")
                time.sleep(3)

    # If we get here, all attempts failed
    raise NoSuchElementException(
        f"Password input not found after {max_attempts} attempts: {str(last_error)}"
    )


def login(driver, email, password):
    """Handle the login process with better error handling and reduced complexity"""
    logger.info("Attempting to log in...")
    time.sleep(10)  # Give more time for initial page load

    # Initial sign in
    sign_in_button = find_sign_in_button(driver)
    if not sign_in_button:
        raise NoSuchElementException("Could not find sign-in button")
    safe_click(driver, sign_in_button, "sign-in button")
    time.sleep(5)  # Wait for auth UI

    # Email sign in
    handle_email_sign_in(driver)

    # Email input and next
    handle_email_input(driver, email)

    # Password input and submit
    handle_password_input(driver, password)


def process_portfolio():
    """Process downloaded portfolio file"""
    # Read the downloaded portfolio
    downloads_path = os.path.expanduser("~/Downloads")
    # Wait longer for the download
    time.sleep(5)

    # Get the most recent csv file
    try:
        if not os.path.isdir(downloads_path):
            logger.error(f"Error: {downloads_path} is not a valid directory")
            return False

        files = [f for f in os.listdir(downloads_path) if f.endswith(".csv")]
        if not files:
            logger.error("No CSV files found in Downloads folder")
            return False

        # Get the latest file path
        latest_filename = max(
            files, key=lambda f: os.path.getctime(os.path.join(downloads_path, f))
        )
        latest_file = os.path.join(downloads_path, latest_filename)

        if not os.path.isfile(latest_file):
            logger.error(f"Error: {latest_file} is not a valid file")
            return False
    except (OSError, IOError) as e:
        logger.error(f"Error accessing files: {str(e)}")
        return False

    # Read the CSV
    df = pd.read_csv(latest_file)

    # Replace crypto tickers
    crypto_mapping = {"BTC": "BTC-USD", "XRP": "XRP-USD", "SOL": "SOL-USD", "ETH": "ETH-USD"}

    # Update tickers if they exist (using 'ticker' column instead of 'Symbol')
    if "ticker" in df.columns:
        logger.info("Found ticker column, updating crypto tickers...")
        df["ticker"] = df["ticker"].replace(crypto_mapping)
        logger.info(
            f"Updated tickers: {df[df['ticker'].isin(crypto_mapping.values())]['ticker'].tolist()}"
        )

        # Fix HK stock tickers with leading zeros
        df["ticker"] = df["ticker"].apply(fix_hk_ticker)
        logger.info("Processed HK tickers with leading zeros")
    else:
        logger.warning("Warning: 'ticker' column not found in CSV")

    # Save to input directory
    os.makedirs(PATHS["INPUT_DIR"], exist_ok=True)
    df.to_csv(FILE_PATHS["PORTFOLIO_FILE"], index=False)
    logger.info(f"Portfolio saved to {FILE_PATHS['PORTFOLIO_FILE']}")

    # Clean up downloaded file
    os.remove(latest_file)
    return True


def handle_cookie_consent(driver):
    """Handle cookie consent if present"""
    try:
        logger.info("Looking for cookie consent...")
        accept_button = wait_and_find_element(
            driver, By.XPATH, "//button[contains(text(), 'Accept All')]"
        )
        if accept_button:
            safe_click(driver, accept_button, "cookie consent")
            time.sleep(2)
    except (NoSuchElementException, TimeoutException):
        logger.info("No cookie consent needed or already accepted")


def handle_portfolio_buttons(driver):
    """Handle clicking portfolio-related buttons"""
    # Click "Load this portfolio" button
    logger.info("Looking for 'Load this portfolio' button...")
    load_button = wait_and_find_element(driver, By.ID, "loadPortfolioStats", timeout=20)
    if not load_button:
        raise NoSuchElementException("Could not find 'Load this portfolio' button")
    safe_click(driver, load_button, "'Load this portfolio' button")
    time.sleep(10)

    # Click "Update" button
    logger.info("Looking for 'Update' button...")
    update_button = wait_and_find_element(driver, By.ID, "updatePi", timeout=20)
    if not update_button:
        raise NoSuchElementException("Could not find 'Update' button")
    safe_click(driver, update_button, "'Update' button")
    time.sleep(10)

    # Click "Export Portfolio" link
    logger.info("Looking for 'Export Portfolio' link...")
    export_link = wait_and_find_element(driver, By.ID, "downloadPortfolio", timeout=20)
    if not export_link:
        raise NoSuchElementException("Could not find 'Export Portfolio' link")
    safe_click(driver, export_link, "'Export Portfolio' link")


async def download_portfolio(provider=None):
    """
    Download portfolio data using eToro API.

    This function:
    1. Fetches portfolio data from eToro API
    2. Retrieves instrument metadata 
    3. Combines and processes the data
    4. Saves it in the expected format for the application

    Args:
        provider: Optional provider instance (not used, but required for compatibility)

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting portfolio download using eToro API...")
    
    # Call the modern eToro API implementation
    return await download_etoro_portfolio(provider)


@with_retry
def download_market_data(
    tickers: List[str],
    include_analyst_data: bool = True,
    include_price_data: bool = True,
    include_financial_data: bool = True,
    provider_name: str = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Download comprehensive market data for a list of tickers.

    This function provides a centralized way to download comprehensive
    market data for multiple tickers in an efficient manner. It leverages
    the provider system to fetch the data.

    Args:
        tickers: List of ticker symbols to download data for
        include_analyst_data: Whether to include analyst recommendations
        include_price_data: Whether to include price and volume data
        include_financial_data: Whether to include financial metrics
        provider_name: Optional provider name to use (defaults to hybrid)

    Returns:
        Dictionary mapping ticker symbols to their market data

    Raises:
        APIError: If there's an error fetching data from the API
        ValidationError: If the input parameters are invalid
    """
    import asyncio

    from yahoofinance import get_provider
    from yahoofinance.core.errors import APIError, YFinanceError
    from yahoofinance.core.logging import get_logger

    logger = get_logger(__name__)

    # Validate input
    if not tickers:
        logger.warning("No tickers provided to download_market_data")
        return {}

    # Initialize provider (async for better performance)
    provider = get_provider(async_api=True, provider_name=provider_name)

    # Define the data download function
    async def fetch_all_data():
        results = {}

        try:
            # Batch fetch ticker info for all tickers
            logger.info(f"Downloading market data for {len(tickers)} tickers")

            # Fetch basic ticker info (always included)
            info_results = await provider.batch_get_ticker_info(tickers)

            # Initialize results dictionary with basic info
            for ticker, info in info_results.items():
                if info:
                    results[ticker] = info
                    # Add source information
                    results[ticker]["data_source"] = getattr(provider, "name", "unknown")

            # Fetch additional data based on flags
            if include_analyst_data:
                logger.info("Fetching analyst data...")
                analyst_results = await provider.batch_get_analyst_data(tickers)
                # Merge analyst data into results
                for ticker, data in analyst_results.items():
                    if ticker in results and data:
                        results[ticker].update(data)

            if include_price_data:
                logger.info("Fetching price data...")
                price_results = await provider.batch_get_price_data(tickers)
                # Merge price data into results
                for ticker, data in price_results.items():
                    if ticker in results and data:
                        results[ticker].update(data)

            if include_financial_data:
                logger.info("Fetching financial data...")
                financial_results = await provider.batch_get_financial_data(tickers)
                # Merge financial data into results
                for ticker, data in financial_results.items():
                    if ticker in results and data:
                        results[ticker].update(data)

            logger.info(f"Successfully downloaded data for {len(results)} tickers")
            return results

        except APIError as e:
            logger.error(f"API error while downloading market data: {str(e)}")
            raise
        except YFinanceError as e:
            logger.error(f"Error downloading market data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading market data: {str(e)}")
            raise APIError(f"Unexpected error downloading market data: {str(e)}")

    # Run the async function
    try:
        return asyncio.run(fetch_all_data())
    except Exception as e:
        logger.error(f"Error in download_market_data: {str(e)}")
        raise


async def fallback_portfolio_download():
    """
    Fallback method that copies the existing portfolio
    from the original location to yahoofinance/input

    Returns:
        bool: True if successful, False otherwise
    """
    # Create a unique ID for this fallback operation
    fallback_id = f"fallback_{int(time.time())}"
    logger.info(f"[{fallback_id}] Starting fallback portfolio download method")

    try:
        # Define source and destination paths
        src_path = FILE_PATHS["PORTFOLIO_FILE"]
        dest_dir = PATHS["INPUT_DIR"]
        dest_path = FILE_PATHS["PORTFOLIO_FILE"]

        logger.info(f"[{fallback_id}] Source path: {src_path}")
        logger.info(f"[{fallback_id}] Destination path: {dest_path}")

        # Ensure destination directory exists
        os.makedirs(dest_dir, exist_ok=True)
        logger.info(f"[{fallback_id}] Ensured destination directory exists: {dest_dir}")

        # Check if source file exists and is readable
        if not os.path.exists(src_path):
            error_msg = f"Source portfolio file not found: {src_path}"
            logger.error(f"[{fallback_id}] {error_msg}")

            # Try to list available files in the input directory
            try:
                input_dir = os.path.dirname(src_path)
                if os.path.exists(input_dir):
                    files = os.listdir(input_dir)
                    logger.info(f"[{fallback_id}] Available files in {input_dir}: {files}")

                    # If any CSV files exist, try to use the most recent one
                    csv_files = [f for f in files if f.endswith(".csv")]
                    if csv_files:
                        most_recent = max(
                            csv_files, key=lambda f: os.path.getmtime(os.path.join(input_dir, f))
                        )
                        alt_src_path = os.path.join(input_dir, most_recent)
                        logger.info(
                            f"[{fallback_id}] Trying to use alternative file: {alt_src_path}"
                        )

                        if os.path.isfile(alt_src_path):
                            src_path = alt_src_path
                            logger.info(
                                f"[{fallback_id}] Using alternative source file: {src_path}"
                            )
                        else:
                            logger.error(
                                f"[{fallback_id}] Alternative file is not valid: {alt_src_path}"
                            )
                            return False
                    else:
                        logger.error(f"[{fallback_id}] No CSV files found in {input_dir}")
                        return False
            except Exception as e:
                logger.error(f"[{fallback_id}] Error checking alternative files: {str(e)}")
                return False

        # Check file size and readability
        try:
            file_size = os.path.getsize(src_path)
            logger.info(f"[{fallback_id}] Source file size: {file_size} bytes")

            if file_size == 0:
                logger.warning(f"[{fallback_id}] Source file is empty: {src_path}")
                # Continue anyway as we'll copy the empty file

            # Try to read the first few lines to verify file is readable
            async with aiofiles.open(src_path, "r", encoding="utf-8") as f:
                header = await f.readline()
                header = header.strip()
                logger.info(f"[{fallback_id}] File header: {header}")
        except Exception as e:
            logger.error(f"[{fallback_id}] Error checking source file: {str(e)}")
            # Continue anyway, as the copy operation might still succeed

        # Copy the file without preserving metadata/permissions for security
        shutil.copy(src_path, dest_path)
        # Set secure permissions: owner read/write, group read, others no access
        import stat
        os.chmod(dest_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
        logger.info(f"[{fallback_id}] Portfolio copied from {src_path} to {dest_path} with secure permissions")

        # Verify the copy was successful
        if os.path.exists(dest_path):
            copy_size = os.path.getsize(dest_path)
            logger.info(f"[{fallback_id}] Copied file size: {copy_size} bytes")

            if copy_size == 0 and file_size > 0:
                logger.error(f"[{fallback_id}] Copy failed: destination file is empty")
                return False

            # Try to read the copied file
            try:
                df = pd.read_csv(dest_path)
                row_count = len(df)
                col_count = len(df.columns)
                logger.info(
                    f"[{fallback_id}] Copied file has {row_count} rows and {col_count} columns"
                )

                # Check for critical columns
                if "ticker" not in df.columns:
                    logger.warning(
                        f"[{fallback_id}] Warning: 'ticker' column missing from portfolio file"
                    )
            except Exception as e:
                logger.error(f"[{fallback_id}] Error reading copied file: {str(e)}")
                # This is not fatal as long as the file was copied

        return True
    except Exception as e:
        logger.error(f"[{fallback_id}] Error in fallback portfolio download: {str(e)}")
        import traceback

        trace = traceback.format_exc()
        logger.error(f"[{fallback_id}] Traceback: {trace}")
        return False


async def download_etoro_portfolio(provider=None):
    """
    Download portfolio data from eToro API.

    This function:
    1. Fetches portfolio data from eToro API
    2. Retrieves instrument metadata 
    3. Combines and processes the data
    4. Saves it in the expected format for the application

    Args:
        provider: Optional provider instance (not used, but required for compatibility)

    Returns:
        bool: True if successful, False otherwise
    """
    import csv
    import json
    import requests
    import uuid
    from collections import defaultdict
    
    # Create a unique run ID for this download attempt
    run_id = f"etoro_download_{int(time.time())}"
    logger.info(f"Starting eToro portfolio download (run ID: {run_id})")

    # Get credentials from environment variables
    username = os.getenv("ETORO_USERNAME", "plessas")
    api_key = os.getenv("ETORO_API_KEY")
    user_key = os.getenv("ETORO_USER_KEY")

    if not api_key or not user_key:
        error_msg = "Error: ETORO_API_KEY and ETORO_USER_KEY must be set in .env file"
        logger.error(f"[{run_id}] {error_msg}")
        print(error_msg)
        return False

    try:
        # Fetch portfolio data
        print("Fetching eToro portfolio data...")
        logger.info(f"[{run_id}] Fetching portfolio for user: {username}")
        
        portfolio = await _fetch_etoro_portfolio(username, api_key, user_key, run_id)
        if not portfolio:
            return False

        # Extract instrument IDs
        positions = portfolio.get("positions", [])
        if not positions:
            print("No positions found in portfolio")
            logger.warning(f"[{run_id}] No positions found in portfolio")
            return False

        instrument_ids = [pos.get("instrumentId") for pos in positions if pos.get("instrumentId")]
        print(f"Found {len(positions)} positions with {len(instrument_ids)} unique instruments")
        logger.info(f"[{run_id}] Found {len(positions)} positions with {len(instrument_ids)} instruments")

        # Fetch instrument metadata
        print("Fetching instrument metadata...")
        metadata = await _fetch_etoro_instrument_metadata(instrument_ids, api_key, user_key, run_id)

        # Combine and process data
        print("Processing portfolio data and fixing ticker formats...")
        processed_data = _process_etoro_portfolio_data(portfolio, metadata, run_id)

        # Save to the expected location
        output_path = os.path.join(PATHS["INPUT_DIR"], "portfolio.csv")
        _save_etoro_portfolio_csv(processed_data, output_path, run_id)

        print(f"eToro portfolio data saved successfully to {output_path}")
        logger.info(f"[{run_id}] eToro portfolio download completed successfully")
        return True

    except Exception as e:
        error_msg = f"Error during eToro portfolio download: {str(e)}"
        logger.error(f"[{run_id}] {error_msg}")
        print(error_msg)
        return False


async def _fetch_etoro_portfolio(username: str, api_key: str, user_key: str, run_id: str):
    """Fetch portfolio data from eToro API."""
    import asyncio
    import aiohttp
    
    url = f"https://www.etoro.com/api/public/v1/user-info/people/{username}/portfolio/live"
    
    headers = {
        "X-REQUEST-ID": str(uuid.uuid4()),
        "X-API-KEY": api_key,
        "X-USER-KEY": user_key,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    for attempt in range(3):
        try:
            # Use aiohttp for async HTTP requests
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    
                    if response.status == 429:
                        wait_time = 2 ** attempt
                        print(f"Rate limited. Waiting {wait_time} seconds...")
                        logger.warning(f"[{run_id}] Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    data = await response.json()
                    logger.info(f"[{run_id}] Successfully fetched portfolio data")
                    return data
            
        except aiohttp.ClientError as e:
            logger.error(f"[{run_id}] HTTP error (attempt {attempt + 1}): {str(e)}")
            if attempt == 2:
                print(f"Failed to fetch portfolio after 3 attempts: {e}")
                return None
            await asyncio.sleep(1)
    
    return None


async def _fetch_etoro_instrument_metadata(instrument_ids: list, api_key: str, user_key: str, run_id: str):
    """Fetch instrument metadata from eToro API."""
    import asyncio
    
    if not instrument_ids:
        return {}

    # Remove duplicates
    unique_ids = list(set(instrument_ids))
    
    # Build URL with comma-separated instrument IDs
    ids_param = ",".join(map(str, unique_ids))
    url = f"https://www.etoro.com/api/public/v1/market-data/instruments?instrumentIds={ids_param}"
    
    headers = {
        "X-REQUEST-ID": str(uuid.uuid4()),
        "X-API-KEY": api_key,
        "X-USER-KEY": user_key,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    
                    if response.status == 429:
                        wait_time = 2 ** attempt
                        print(f"Rate limited. Waiting {wait_time} seconds...")
                        logger.warning(f"[{run_id}] Metadata rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Convert to dictionary for easy lookup
                    metadata_dict = {}
                    if "instrumentDisplayDatas" in data:
                        for instrument in data["instrumentDisplayDatas"]:
                            if "instrumentID" in instrument:
                                metadata_dict[instrument["instrumentID"]] = instrument
                    
                    logger.info(f"[{run_id}] Successfully fetched metadata for {len(metadata_dict)} instruments")
                    return metadata_dict
            
        except aiohttp.ClientError as e:
            logger.error(f"[{run_id}] Metadata HTTP error (attempt {attempt + 1}): {str(e)}")
            if attempt == 2:
                print(f"Failed to fetch metadata after 3 attempts: {e}")
                return {}
            await asyncio.sleep(1)
    
    return {}


def _fix_yahoo_ticker_format(ticker: str) -> str:
    """
    Convert ticker to Yahoo Finance format.
    
    Rules:
    - HK tickers: Format to have exactly 4 digits (e.g., 1.HK -> 0001.HK, 700.HK -> 0700.HK)
    - Cryptocurrencies: BTC/ETH -> BTC-USD/ETH-USD
    - Options/Futures formatting with ^ and = symbols as needed
    - All other tickers remain unchanged
    """
    # Backup original ticker
    original = ticker
    
    # Handle Hong Kong tickers (ensure 4 digits with leading zeros)
    if ticker.endswith('.HK'):
        numeric_part = ticker.split('.')[0]
        try:
            # First remove any leading zeros
            cleaned_numeric = numeric_part.lstrip('0')
            # If empty, this was all zeros
            if not cleaned_numeric:
                cleaned_numeric = '0'
            # Now format to have exactly 4 digits with leading zeros
            formatted_numeric = cleaned_numeric.zfill(4)
            ticker = f"{formatted_numeric}.HK"
        except Exception:
            # If any error occurs, keep original
            ticker = original
    
    # Handle cryptocurrencies (add -USD suffix if not present)
    elif ticker in ('BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK', 'XLM', 'DOGE'):
        ticker = f"{ticker}-USD"
    
    # Handle European tickers that need US ticker mapping
    elif ticker == 'ASML.NV':
        ticker = 'ASML'
    
    # Handle special VIX futures like VIX.MAY25
    elif ticker.startswith('VIX.'):
        # Extract month and year
        try:
            parts = ticker.split('.')
            if len(parts) == 2 and len(parts[1]) >= 5:
                month = parts[1][:3]
                year = parts[1][3:]
                
                # Convert to Yahoo Finance format
                ticker = f"^VIX{month}{year}"
        except Exception:
            # If any error occurs, keep original
            ticker = original
    
    # Return the fixed ticker
    return ticker


def _process_etoro_portfolio_data(portfolio: dict, metadata: dict, run_id: str):
    """Process eToro portfolio data into the format expected by the application."""
    from collections import defaultdict
    
    # Group positions by symbol (using original symbol first)
    grouped = defaultdict(list)
    
    for position in portfolio.get("positions", []):
        instrument_id = position.get("instrumentId")
        instrument_meta = metadata.get(instrument_id, {})
        
        # Get symbol, defaulting to Unknown if not available
        symbol = instrument_meta.get("symbolFull", "Unknown")
        if symbol and symbol != "Unknown":
            grouped[symbol].append({
                "position": position,
                "metadata": instrument_meta
            })

    # Process grouped data
    processed_data = []
    
    for symbol, symbol_positions in grouped.items():
        # Get data from first position
        first_pos_data = symbol_positions[0]
        first_pos = first_pos_data["position"]
        first_meta = first_pos_data["metadata"]
        
        # Calculate aggregated values
        num_positions = len(symbol_positions)
        total_investment_pct = sum(p["position"].get("investmentPct", 0) for p in symbol_positions)
        total_net_profit = sum(p["position"].get("netProfit", 0) for p in symbol_positions)
        
        # Calculate weighted average open rate
        weighted_open_rate = 0
        if total_investment_pct > 0:
            weighted_open_rate = sum(
                p["position"].get("openRate", 0) * p["position"].get("investmentPct", 0) 
                for p in symbol_positions
            ) / total_investment_pct
        
        # Get earliest open timestamp
        earliest_open = min(p["position"].get("openTimestamp", "") for p in symbol_positions)
        
        # Check if all positions are buy
        all_buy = all(p["position"].get("isBuy", True) for p in symbol_positions)
        
        # Get leverage (should be same for all positions of same symbol)
        leverages = set(p["position"].get("leverage", 1) for p in symbol_positions)
        leverage = leverages.pop() if len(leverages) == 1 else max(leverages)
        
        # Fix the ticker format for Yahoo Finance compatibility
        fixed_symbol = _fix_yahoo_ticker_format(symbol)
        
        # Log ticker changes if any
        if fixed_symbol != symbol:
            logger.info(f"[{run_id}] Fixed ticker: {symbol} -> {fixed_symbol}")
        
        # Create the processed row
        processed_row = {
            "symbol": fixed_symbol,  # Use the fixed symbol
            "instrumentDisplayName": first_meta.get("instrumentDisplayName", "Unknown"),
            "instrumentId": first_pos.get("instrumentId", ""),
            "numPositions": num_positions,
            "totalInvestmentPct": round(total_investment_pct, 6),
            "avgOpenRate": round(weighted_open_rate, 4),
            "totalNetProfit": round(total_net_profit, 3),
            "totalNetProfitPct": round((total_net_profit / total_investment_pct) if total_investment_pct > 0 else 0, 3),
            "earliestOpenTimestamp": earliest_open,
            "isBuy": all_buy,
            "leverage": leverage,
            "instrumentTypeId": first_meta.get("instrumentTypeID", ""),
            "exchangeId": first_meta.get("exchangeID", ""),
            "exchangeName": first_meta.get("priceSource", ""),
            "stocksIndustryId": first_meta.get("stocksIndustryID", ""),
            "isInternalInstrument": first_meta.get("isInternalInstrument", False)
        }
        
        processed_data.append(processed_row)
    
    # Sort by total investment percentage descending
    processed_data.sort(key=lambda x: x["totalInvestmentPct"], reverse=True)
    
    logger.info(f"[{run_id}] Processed {len(processed_data)} unique symbols")
    return processed_data


def _save_etoro_portfolio_csv(data: list, output_path: str, run_id: str):
    """Save eToro portfolio data to CSV in the expected format."""
    import csv
    
    if not data:
        print("No data to save")
        logger.warning(f"[{run_id}] No data to save")
        return

    # Define the CSV columns to match the expected format
    fieldnames = [
        "symbol",
        "instrumentDisplayName", 
        "instrumentId",
        "numPositions",
        "totalInvestmentPct",
        "avgOpenRate",
        "totalNetProfit",
        "totalNetProfitPct",
        "earliestOpenTimestamp",
        "isBuy",
        "leverage",
        "instrumentTypeId",
        "exchangeId", 
        "exchangeName",
        "stocksIndustryId",
        "isInternalInstrument"
    ]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    logger.info(f"[{run_id}] eToro portfolio data saved to {output_path}")
    
    # Print summary
    total_investment = sum(row['totalInvestmentPct'] for row in data)
    total_profit = sum(row['totalNetProfit'] for row in data)
    print(f"\neToro Portfolio Summary:")
    print(f"Total unique symbols: {len(data)}")
    print(f"Total investment: {total_investment:.2f}%")
    print(f"Total net profit: {total_profit:.2f}")
    print("Note: Ticker formats have been automatically fixed for Yahoo Finance compatibility")


# Test function
if __name__ == "__main__":
    import asyncio

    asyncio.run(download_portfolio())
