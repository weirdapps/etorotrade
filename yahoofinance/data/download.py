"""
Data downloading utilities.

This module provides functions for downloading data from various sources.
"""

import os
import pandas as pd
import time
import logging
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

logger = logging.getLogger(__name__)

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
        raise
    except WebDriverException as e:
        logger.error(f"WebDriver error clicking {description}: {str(e)}")
        raise

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
    except Exception as e:
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
        raise
    except WebDriverException as e:
        logger.error(f"WebDriver error finding sign-in button: {str(e)}")
        raise

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
    time.sleep(1)
    password_input.clear()
    password_input.send_keys(password)
    time.sleep(2)

    # Try to find sign in button
    sign_in_button = wait_and_find_element(
        driver,
        By.XPATH,
        "//button[contains(text(), 'Sign in') or contains(text(), 'SIGN IN')]",
        timeout=5
    )
    if sign_in_button:
        safe_click(driver, sign_in_button, "Sign in button")
    else:
        logger.info("No Sign in button found, submitting form...")
        password_input.send_keys(Keys.RETURN)
        logger.info("Sent RETURN key to password input")

    time.sleep(10)  # Wait longer for login to complete

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

def download_portfolio():
    """
    Download portfolio data from pi-screener.com.
    
    This function automates the process of:
    1. Logging into pi-screener.com
    2. Navigating to the portfolio page
    3. Downloading the portfolio data
    4. Processing the downloaded file to standardize tickers
    5. Saving the processed file to yahoofinance/input
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Get credentials from environment variables
    pi_screener_email = os.getenv('PI_SCREENER_EMAIL')
    pi_screener_password = os.getenv('PI_SCREENER_PASSWORD')

    if not pi_screener_email or not pi_screener_password:
        logger.error("Error: PI_SCREENER_EMAIL and PI_SCREENER_PASSWORD must be set in .env file")
        print("Error: PI_SCREENER_EMAIL and PI_SCREENER_PASSWORD must be set in .env file")
        return False

    # Check if selenium is available
    try:
        import selenium
    except ImportError:
        logger.error("Selenium is not installed. Using fallback method.")
        print("Selenium is not installed. Using fallback method.")
        return fallback_portfolio_download()

    driver = None
    try:
        # Setup and navigate
        logger.info("Setting up Chrome driver...")
        print("Setting up Chrome driver...")
        driver = setup_driver()
        
        logger.info("Navigating to pi-screener.com...")
        print("Navigating to pi-screener.com...")
        driver.get("https://app.pi-screener.com/")
        time.sleep(10)  # Wait longer for initial page load
        
        # Handle initial page setup
        handle_cookie_consent(driver)
        
        print("Logging in to pi-screener.com...")
        login(driver, pi_screener_email, pi_screener_password)
        
        # Wait for page to load after login
        logger.info("Waiting for page to load after login...")
        print("Waiting for page to load after login...")
        time.sleep(20)
        
        # Handle portfolio operations
        print("Accessing portfolio...")
        handle_portfolio_buttons(driver)
        
        # Process download
        logger.info("Waiting for download...")
        print("Waiting for download...")
        time.sleep(5)
        if process_portfolio():
            print("Portfolio successfully downloaded and processed.")
            return True
        else:
            print("Failed to process downloaded portfolio.")
            return fallback_portfolio_download()
        
    except (NoSuchElementException, ElementClickInterceptedException) as e:
        logger.error(f"Element interaction error: {str(e)}")
        print(f"Error interacting with web elements: {str(e)}")
        return fallback_portfolio_download()
    except WebDriverException as e:
        logger.error(f"WebDriver error: {str(e)}")
        print(f"WebDriver error: {str(e)}")
        return fallback_portfolio_download()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return fallback_portfolio_download()
    finally:
        if driver:
            logger.info("Closing Chrome...")
            print("Closing Chrome...")
            driver.quit()

def fallback_portfolio_download():
    """
    Fallback method that copies the existing portfolio 
    from the original location to yahoofinance/input
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("Using fallback portfolio download method...")
        # Define source and destination paths
        src_path = FILE_PATHS["PORTFOLIO_FILE"]
        dest_dir = PATHS["INPUT_DIR"]
        dest_path = FILE_PATHS["PORTFOLIO_FILE"]
        
        # Ensure destination directory exists
        os.makedirs(dest_dir, exist_ok=True)
        
        # Check if source file exists
        if not os.path.exists(src_path):
            logger.error(f"Source portfolio file not found: {src_path}")
            print(f"Source portfolio file not found: {src_path}")
            return False
        
        # Copy the file
        shutil.copy2(src_path, dest_path)
        logger.info(f"Portfolio copied from {src_path} to {dest_path}")
        print(f"Portfolio copied from {src_path} to {dest_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error in fallback portfolio download: {str(e)}")
        print(f"Error in fallback portfolio download: {str(e)}")
        return False

# Test function
if __name__ == "__main__":
    download_portfolio()