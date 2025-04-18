"""
Data downloading utilities.

This module provides functions for downloading data from various sources.
"""

import os
from typing import List, Dict, Any, Optional, Union

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation
import pandas as pd
import time
from ..core.logging_config import get_logger
import shutil
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    WebDriverException
)

from ..core.config import PATHS, FILE_PATHS

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
    if not isinstance(ticker, str) or not ticker.endswith('.HK'):
        return ticker
        
    parts = ticker.split('.')
    if len(parts) != 2:
        return ticker
        
    numeric_part = parts[0]
    fixed_ticker = ticker
    
    # Process based on digit count
    if len(numeric_part) < 4:
        # Add leading zeros for fewer than 4 digits
        fixed_ticker = numeric_part.zfill(4) + '.HK'
    elif len(numeric_part) > 4 and numeric_part.startswith('0'):
        # Remove leading zeros for more than 4 digits
        stripped_part = numeric_part.lstrip('0')
        fixed_ticker = (stripped_part.zfill(4) if len(stripped_part) < 4 else stripped_part) + '.HK'
    
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
    """Setup Chrome WebDriver with appropriate options"""
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Add additional options to help with stability
    options.add_argument('--disable-web-security')
    options.add_argument('--disable-features=IsolateOrigins,site-per-process')
    # Add options for slow network
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1200,800')
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
    except YFinanceError as e:
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
    email_sign_in = wait_and_find_element(driver, By.XPATH, "//button[contains(., 'Sign in with email')]")
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
    next_button = wait_and_find_element(driver, By.XPATH, "//button[contains(text(), 'Next') or contains(text(), 'NEXT')]")
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
        (By.XPATH, "//input[contains(@class, 'password')]")
    ]:
        try:
            password_input = wait_and_find_element(driver, selector[0], selector[1], timeout=timeout)
            if password_input:
                return password_input
        except (NoSuchElementException, TimeoutException):
            continue
    return None

def handle_password_submit(driver, password_input, password):
    """Handle password submission and final sign in"""
    logger.info("Found password input, entering password...")
    print("DEBUG: Found password input, entering password...")
    
    # Take a screenshot before password entry to help with debugging
    try:
        screenshot_path = os.path.expanduser("~/Downloads/login_before_password.png")
        driver.save_screenshot(screenshot_path)
        print(f"DEBUG: Saved screenshot before password entry to {screenshot_path}")
    except Exception as e:
        print(f"DEBUG: Failed to save screenshot: {str(e)}")
    
    # Clear any existing text in the password field
    time.sleep(1)
    password_input.clear()
    
    # Enter password
    password_input.send_keys(password)
    print(f"DEBUG: Entered password (length: {len(password)})")
    time.sleep(2)

    # Try multiple different selectors for the sign-in button
    print("DEBUG: Looking for sign-in button after password entry...")
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
    
    # Take a screenshot to help with debugging
    try:
        screenshot_path = os.path.expanduser("~/Downloads/login_after_password.png")
        driver.save_screenshot(screenshot_path)
        print(f"DEBUG: Saved screenshot after password entry to {screenshot_path}")
    except Exception as e:
        print(f"DEBUG: Failed to save screenshot: {str(e)}")
    
    # Try each selector
    for selector_type, selector in button_selectors:
        try:
            print(f"DEBUG: Trying selector: {selector}")
            button = driver.find_element(selector_type, selector)
            if button:
                sign_in_button = button
                print(f"DEBUG: Found sign-in button with selector: {selector}")
                break
        except NoSuchElementException:
            continue
    
    # Click the sign-in button if found
    if sign_in_button:
        print("DEBUG: Attempting to click sign-in button")
        try:
            # First try with safe_click
            safe_click(driver, sign_in_button, "Sign in button")
        except Exception as e:
            print(f"DEBUG: Error with safe_click: {str(e)}, trying direct click")
            try:
                # Then try direct click if JavaScript click fails
                sign_in_button.click()
                print("DEBUG: Direct click succeeded")
            except Exception as e2:
                print(f"DEBUG: Direct click failed: {str(e2)}, trying Enter key")
                password_input.send_keys(Keys.RETURN)
    else:
        # Try looking for a form instead and submit it
        print("DEBUG: No sign-in button found, looking for form to submit...")
        try:
            form = driver.find_element(By.TAG_NAME, "form")
            if form:
                print("DEBUG: Found form, submitting with JavaScript")
                try:
                    driver.execute_script("arguments[0].submit();", form)
                    print("DEBUG: Form submitted successfully")
                except Exception as e:
                    print(f"DEBUG: Error submitting form with JavaScript: {str(e)}")
                    print("DEBUG: Trying to submit with Enter key instead")
                    password_input.send_keys(Keys.RETURN)
            else:
                print("DEBUG: No form found, submitting with Enter key")
                password_input.send_keys(Keys.RETURN)
        except NoSuchElementException:
            print("DEBUG: No form found, submitting with Enter key")
            password_input.send_keys(Keys.RETURN)

    print("DEBUG: Waiting for login to complete...")
    # Take a final screenshot to verify login status
    try:
        time.sleep(3)  # Short wait for UI update
        screenshot_path = os.path.expanduser("~/Downloads/login_after_submit.png")
        driver.save_screenshot(screenshot_path)
        print(f"DEBUG: Saved screenshot after submission to {screenshot_path}")
    except Exception as e:
        print(f"DEBUG: Failed to save screenshot: {str(e)}")
    
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
        except (NoSuchElementException, TimeoutException, ElementClickInterceptedException, WebDriverException) as e:
            last_error = e
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_attempts - 1:
                logger.info("Retrying...")
                time.sleep(3)

    # If we get here, all attempts failed
    raise NoSuchElementException(f"Password input not found after {max_attempts} attempts: {str(last_error)}")

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

        files = [f for f in os.listdir(downloads_path) if f.endswith('.csv')]
        if not files:
            logger.error("No CSV files found in Downloads folder")
            return False
        
        # Get the latest file path
        latest_filename = max(files, key=lambda f: os.path.getctime(os.path.join(downloads_path, f)))
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
    crypto_mapping = {
        'BTC': 'BTC-USD',
        'XRP': 'XRP-USD',
        'SOL': 'SOL-USD',
        'ETH': 'ETH-USD'
    }
    
    # Update tickers if they exist (using 'ticker' column instead of 'Symbol')
    if 'ticker' in df.columns:
        logger.info("Found ticker column, updating crypto tickers...")
        df['ticker'] = df['ticker'].replace(crypto_mapping)
        logger.info(f"Updated tickers: {df[df['ticker'].isin(crypto_mapping.values())]['ticker'].tolist()}")
        
        # Fix HK stock tickers with leading zeros
        df['ticker'] = df['ticker'].apply(fix_hk_ticker)
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
        accept_button = wait_and_find_element(driver, By.XPATH, "//button[contains(text(), 'Accept All')]")
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
    Download portfolio data from pi-screener.com.
    
    This function automates the process of:
    1. Logging into pi-screener.com
    2. Navigating to the portfolio page
    3. Downloading the portfolio data
    4. Processing the downloaded file to standardize tickers
    5. Saving the processed file to yahoofinance/input
    
    Args:
        provider: Optional provider instance (not used, but required for compatibility)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("DEBUG: Inside download_portfolio function")
    
    # Create a unique run ID for this download attempt to track related logs
    run_id = f"download_{int(time.time())}"
    print(f"DEBUG: Download run ID: {run_id}")
    logger.info(f"Starting portfolio download (run ID: {run_id})")
    
    # Get credentials from environment variables
    pi_screener_email = os.getenv('PI_SCREENER_EMAIL')
    pi_screener_password = os.getenv('PI_SCREENER_PASSWORD')

    if not pi_screener_email or not pi_screener_password:
        error_msg = "Error: PI_SCREENER_EMAIL and PI_SCREENER_PASSWORD must be set in .env file"
        logger.error(f"[{run_id}] {error_msg}")
        print(error_msg)
        return False

    # Check if selenium is available
    try:
        import selenium
        logger.info(f"[{run_id}] Selenium is available (version: {selenium.__version__})")
    except ImportError:
        logger.error(f"[{run_id}] Selenium is not installed. Using fallback method.")
        print("Selenium is not installed. Using fallback method.")
        return await fallback_portfolio_download()

    driver = None
    try:
        # Setup and navigate
        logger.info(f"[{run_id}] Setting up Chrome driver...")
        print("Setting up Chrome driver...")
        driver = setup_driver()
        
        # Take a screenshot of the browser right after setup
        try:
            screenshot_path = os.path.expanduser(f"~/Downloads/chrome_setup_{run_id}.png")
            driver.save_screenshot(screenshot_path)
            logger.info(f"[{run_id}] Saved initial browser screenshot to {screenshot_path}")
        except Exception as e:
            logger.warning(f"[{run_id}] Failed to save initial screenshot: {str(e)}")
        
        logger.info(f"[{run_id}] Navigating to pi-screener.com...")
        print("Navigating to pi-screener.com...")
        driver.get("https://app.pi-screener.com/")
        time.sleep(10)  # Wait longer for initial page load
        
        # Take a screenshot after initial page load
        try:
            screenshot_path = os.path.expanduser(f"~/Downloads/initial_page_{run_id}.png")
            driver.save_screenshot(screenshot_path)
            logger.info(f"[{run_id}] Saved page load screenshot to {screenshot_path}")
        except Exception as e:
            logger.warning(f"[{run_id}] Failed to save page screenshot: {str(e)}")
        
        # Handle initial page setup
        handle_cookie_consent(driver)
        
        print("Logging in to pi-screener.com...")
        logger.info(f"[{run_id}] Attempting to log in with email: {pi_screener_email[:3]}***")
        login(driver, pi_screener_email, pi_screener_password)
        
        # Wait for page to load after login
        logger.info(f"[{run_id}] Waiting for page to load after login...")
        print("Waiting for page to load after login...")
        time.sleep(20)
        
        # Take a screenshot after login
        try:
            screenshot_path = os.path.expanduser(f"~/Downloads/post_login_{run_id}.png")
            driver.save_screenshot(screenshot_path)
            logger.info(f"[{run_id}] Saved post-login screenshot to {screenshot_path}")
        except Exception as e:
            logger.warning(f"[{run_id}] Failed to save post-login screenshot: {str(e)}")
        
        # Check if login was successful
        current_url = driver.current_url
        logger.info(f"[{run_id}] Current URL after login: {current_url}")
        print(f"DEBUG: Current URL after login: {current_url}")
        
        if "login" in current_url.lower() or "signin" in current_url.lower() or "auth" in current_url.lower():
            logger.error(f"[{run_id}] Login appears to have failed. Still on login page: {current_url}")
            print("DEBUG: Login appears to have failed. Still on login page.")
            return await fallback_portfolio_download()
        
        # Handle portfolio operations
        print("Accessing portfolio...")
        logger.info(f"[{run_id}] Accessing portfolio features...")
        handle_portfolio_buttons(driver)
        
        # Process download
        logger.info(f"[{run_id}] Waiting for download...")
        print("Waiting for download...")
        time.sleep(5)
        
        if process_portfolio():
            success_msg = "Portfolio successfully downloaded and processed."
            logger.info(f"[{run_id}] {success_msg}")
            print(success_msg)
            return True
        else:
            failure_msg = "Failed to process downloaded portfolio."
            logger.warning(f"[{run_id}] {failure_msg} Trying fallback method.")
            print(failure_msg)
            return await fallback_portfolio_download()
        
    except NoSuchElementException as e:
        error_msg = f"Element not found: {str(e)}"
        logger.error(f"[{run_id}] {error_msg}")
        print(f"Error: {error_msg}")
        # Save DOM page source to help debugging
        if driver:
            try:
                page_source_path = os.path.expanduser(f"~/Downloads/page_source_{run_id}.html")
                with open(page_source_path, 'w') as f:
                    f.write(driver.page_source)
                logger.info(f"[{run_id}] Saved page source to {page_source_path}")
            except Exception as e:
                logger.warning(f"[{run_id}] Failed to save page source: {str(e)}")
        return await fallback_portfolio_download()
    
    except ElementClickInterceptedException as e:
        error_msg = f"Click intercepted: {str(e)}"
        logger.error(f"[{run_id}] {error_msg}")
        print(f"Error: {error_msg}")
        # Take a screenshot of the page where the click was intercepted
        if driver:
            try:
                screenshot_path = os.path.expanduser(f"~/Downloads/click_intercepted_{run_id}.png")
                driver.save_screenshot(screenshot_path)
                logger.info(f"[{run_id}] Saved click interception screenshot to {screenshot_path}")
            except Exception as e:
                logger.warning(f"[{run_id}] Failed to save screenshot: {str(e)}")
        return await fallback_portfolio_download()
    
    except WebDriverException as e:
        error_msg = f"WebDriver error: {str(e)}"
        logger.error(f"[{run_id}] {error_msg}")
        print(f"Error: {error_msg}")
        return await fallback_portfolio_download()
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[{run_id}] {error_msg}")
        print(f"Error: {error_msg}")
        import traceback
        trace = traceback.format_exc()
        logger.error(f"[{run_id}] Traceback: {trace}")
        return await fallback_portfolio_download()
    
    finally:
        if driver:
            logger.info(f"[{run_id}] Closing Chrome...")
            print("Closing Chrome...")
            try:
                driver.quit()
                logger.info(f"[{run_id}] Chrome closed successfully")
            except Exception as e:
                logger.warning(f"[{run_id}] Error while closing Chrome: {str(e)}")
                print(f"Warning: Error while closing Chrome: {str(e)}")

@with_retry
def download_market_data(
    tickers: List[str], 
    include_analyst_data: bool = True,
    include_price_data: bool = True,
    include_financial_data: bool = True,
    provider_name: str = None
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
    from yahoofinance import get_provider
    import asyncio
    from yahoofinance.core.logging_config import get_logger
    from yahoofinance.core.errors import YFinanceError, APIError
    
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
                    results[ticker]['data_source'] = getattr(provider, 'name', 'unknown')
            
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
    print("Using fallback portfolio download method...")
    
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
            print(error_msg)
            
            # Try to list available files in the input directory
            try:
                input_dir = os.path.dirname(src_path)
                if os.path.exists(input_dir):
                    files = os.listdir(input_dir)
                    logger.info(f"[{fallback_id}] Available files in {input_dir}: {files}")
                    print(f"Available files in {input_dir}: {files}")
                    
                    # If any CSV files exist, try to use the most recent one
                    csv_files = [f for f in files if f.endswith('.csv')]
                    if csv_files:
                        most_recent = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(input_dir, f)))
                        alt_src_path = os.path.join(input_dir, most_recent)
                        logger.info(f"[{fallback_id}] Trying to use alternative file: {alt_src_path}")
                        print(f"Trying to use alternative file: {most_recent}")
                        
                        if os.path.isfile(alt_src_path):
                            src_path = alt_src_path
                            logger.info(f"[{fallback_id}] Using alternative source file: {src_path}")
                            print(f"Using alternative source file: {most_recent}")
                        else:
                            logger.error(f"[{fallback_id}] Alternative file is not valid: {alt_src_path}")
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
                print(f"Warning: Source file is empty: {src_path}")
                # Continue anyway as we'll copy the empty file
            
            # Try to read the first few lines to verify file is readable
            with open(src_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                logger.info(f"[{fallback_id}] File header: {header}")
        except Exception as e:
            logger.error(f"[{fallback_id}] Error checking source file: {str(e)}")
            print(f"Error checking source file: {str(e)}")
            # Continue anyway, as the copy operation might still succeed
        
        # Copy the file
        shutil.copy2(src_path, dest_path)
        logger.info(f"[{fallback_id}] Portfolio copied from {src_path} to {dest_path}")
        print(f"Portfolio copied from {src_path} to {dest_path}")
        
        # Verify the copy was successful
        if os.path.exists(dest_path):
            copy_size = os.path.getsize(dest_path)
            logger.info(f"[{fallback_id}] Copied file size: {copy_size} bytes")
            
            if copy_size == 0 and file_size > 0:
                logger.error(f"[{fallback_id}] Copy failed: destination file is empty")
                print("Error: Copy failed - destination file is empty")
                return False
            
            # Try to read the copied file
            try:
                df = pd.read_csv(dest_path)
                row_count = len(df)
                col_count = len(df.columns)
                logger.info(f"[{fallback_id}] Copied file has {row_count} rows and {col_count} columns")
                print(f"Copied portfolio file has {row_count} rows and {col_count} columns")
                
                # Check for critical columns
                if 'ticker' not in df.columns:
                    logger.warning(f"[{fallback_id}] Warning: 'ticker' column missing from portfolio file")
                    print("Warning: 'ticker' column missing from portfolio file")
            except Exception as e:
                logger.error(f"[{fallback_id}] Error reading copied file: {str(e)}")
                print(f"Error reading copied file: {str(e)}")
                # This is not fatal as long as the file was copied
        
        return True
    except Exception as e:
        logger.error(f"[{fallback_id}] Error in fallback portfolio download: {str(e)}")
        print(f"Error in fallback portfolio download: {str(e)}")
        import traceback
        trace = traceback.format_exc()
        logger.error(f"[{fallback_id}] Traceback: {trace}")
        return False

# Test function
if __name__ == "__main__":
    import asyncio
    asyncio.run(download_portfolio())