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
import time
import os
import pandas as pd

def safe_click(driver, element, description="element"):
    """Helper function to safely click an element using JavaScript"""
    try:
        time.sleep(1)  # Brief pause before clicking
        driver.execute_script("arguments[0].click();", element)
        print(f"Clicked {description}")
        return True
    except ElementClickInterceptedException as e:
        print(f"Click intercepted for {description}: {str(e)}")
        raise
    except WebDriverException as e:
        print(f"WebDriver error clicking {description}: {str(e)}")
        raise

def setup_driver():
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
        print(f"Error finding element {value}: {str(e)}")
        return None

def find_sign_in_button(driver):
    """Helper function to find and click the sign-in button"""
    try:
        print("Looking for sign-in button...")
        sign_in_elements = driver.find_elements(By.CLASS_NAME, "action-button")
        for element in sign_in_elements:
            if "Sign in" in element.text:
                print(f"Found sign-in button with text: {element.text}")
                return element
        return None
    except NoSuchElementException as e:
        print(f"Sign-in button not found: {str(e)}")
        raise
    except WebDriverException as e:
        print(f"WebDriver error finding sign-in button: {str(e)}")
        raise

def login(driver, email, password):
    """Handle the login process with better error handling and reduced complexity"""
    print("Attempting to log in...")
    time.sleep(10)  # Give more time for initial page load
    
    # Find and click the sign-in button
    sign_in_button = find_sign_in_button(driver)
    if not sign_in_button:
        raise NoSuchElementException("Could not find sign-in button")
    safe_click(driver, sign_in_button, "sign-in button")

    time.sleep(5)  # Wait longer for auth UI

    def handle_email_sign_in(driver):
        """Handle the email sign-in process"""
        print("Looking for email sign-in button...")
        email_sign_in = wait_and_find_element(driver, By.XPATH, "//button[contains(., 'Sign in with email')]")
        if not email_sign_in:
            raise NoSuchElementException("Email sign-in button not found")
        safe_click(driver, email_sign_in, "email sign-in button")
        time.sleep(5)

    try:
        handle_email_sign_in(driver)
    except (NoSuchElementException, ElementClickInterceptedException, WebDriverException) as e:
        print(f"Error with email sign-in: {str(e)}")
        raise

    def handle_email_input(driver, email):
        """Handle the email input and next button process"""
        print("Looking for email input...")
        email_input = wait_and_find_element(driver, By.ID, "ui-sign-in-email-input")
        if not email_input:
            email_input = wait_and_find_element(driver, By.XPATH, "//input[@type='email']")
        if not email_input:
            raise NoSuchElementException("Email input field not found")

        print("Entering email...")
        email_input.clear()
        time.sleep(1)
        email_input.send_keys(email)
        time.sleep(2)

        # Look for the Next button
        print("Looking for Next button...")
        next_button = wait_and_find_element(driver, By.XPATH, "//button[contains(text(), 'Next') or contains(text(), 'NEXT')]")
        if next_button:
            safe_click(driver, next_button, "Next button")
        else:
            print("No Next button found, submitting form...")
            email_input.send_keys(Keys.RETURN)
            print("Sent RETURN key to email input")

        time.sleep(5)  # Wait longer for password field

    try:
        handle_email_input(driver, email)
    except (NoSuchElementException, ElementClickInterceptedException, WebDriverException) as e:
        print(f"Error with email input process: {str(e)}")
        raise

    def handle_password_input(driver, password, max_attempts=3):
        """Handle the password input and final sign in process"""
        print("Looking for password input...")
        password_input = None
        last_error = None

        for attempt in range(max_attempts):
            try:
                # Try different selectors to find password input
                for selector in [
                    (By.XPATH, "//input[@type='password']"),
                    (By.CSS_SELECTOR, "input[type='password']"),
                    (By.XPATH, "//input[contains(@class, 'password')]")
                ]:
                    try:
                        password_input = wait_and_find_element(driver, selector[0], selector[1], timeout=5)
                        if password_input:
                            break
                    except (NoSuchElementException, TimeoutException):
                        continue

                if password_input:
                    print("Found password input, entering password...")
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
                        print("No Sign in button found, submitting form...")
                        password_input.send_keys(Keys.RETURN)
                        print("Sent RETURN key to password input")

                    time.sleep(10)  # Wait longer for login to complete
                    return True

            except (NoSuchElementException, TimeoutException, ElementClickInterceptedException, WebDriverException) as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                    time.sleep(3)

        # If we get here, all attempts failed
        raise NoSuchElementException(f"Password input not found after {max_attempts} attempts: {str(last_error)}")

    try:
        handle_password_input(driver, password)
    except (NoSuchElementException, ElementClickInterceptedException, WebDriverException) as e:
        print(f"Error with password input process: {str(e)}")
        raise

def process_portfolio():
    # Read the downloaded portfolio
    downloads_path = os.path.expanduser("~/Downloads")
    # Wait longer for the download
    time.sleep(5)
    
    # Get the most recent csv file
    try:
        if not os.path.isdir(downloads_path):
            print(f"Error: {downloads_path} is not a valid directory")
            return

        files = [f for f in os.listdir(downloads_path) if f.endswith('.csv')]
        if not files:
            print("No CSV files found in Downloads folder")
            return
        
        # Get the latest file path
        latest_filename = max(files, key=lambda f: os.path.getctime(os.path.join(downloads_path, f)))
        latest_file = os.path.join(downloads_path, latest_filename)
        
        if not os.path.isfile(latest_file):
            print(f"Error: {latest_file} is not a valid file")
            return
    except (OSError, IOError) as e:
        print(f"Error accessing files: {str(e)}")
        return
    
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
        print("Found ticker column, updating crypto tickers...")
        df['ticker'] = df['ticker'].replace(crypto_mapping)
        print("Updated tickers:", df[df['ticker'].isin(crypto_mapping.values())]['ticker'].tolist())
    else:
        print("Warning: 'ticker' column not found in CSV")
    
    # Save to input directory
    output_dir = os.path.join(os.path.dirname(__file__), 'input')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'portfolio.csv')
    df.to_csv(output_path, index=False)
    print(f"Portfolio saved to {output_path}")
    
    # Clean up downloaded file
    os.remove(latest_file)

def download_portfolio():
    """Main function to download and process the portfolio"""
    driver = None
    try:
        print("Setting up Chrome driver...")
        driver = setup_driver()
        
        print("Navigating to pi-screener.com...")
        driver.get("https://app.pi-screener.com/")
        time.sleep(10)  # Wait longer for initial page load
        
        def handle_cookie_consent(driver):
            """Handle cookie consent if present"""
            try:
                print("Looking for cookie consent...")
                accept_button = wait_and_find_element(driver, By.XPATH, "//button[contains(text(), 'Accept All')]")
                if accept_button:
                    safe_click(driver, accept_button, "cookie consent")
                    time.sleep(2)
            except (NoSuchElementException, TimeoutException):
                print("No cookie consent needed or already accepted")

        def handle_portfolio_buttons(driver):
            """Handle clicking portfolio-related buttons"""
            # Click "Load this portfolio" button
            print("Looking for 'Load this portfolio' button...")
            load_button = wait_and_find_element(driver, By.ID, "loadPortfolioStats", timeout=20)
            if not load_button:
                raise NoSuchElementException("Could not find 'Load this portfolio' button")
            safe_click(driver, load_button, "'Load this portfolio' button")
            time.sleep(10)

            # Click "Update" button
            print("Looking for 'Update' button...")
            update_button = wait_and_find_element(driver, By.ID, "updatePi", timeout=20)
            if not update_button:
                raise NoSuchElementException("Could not find 'Update' button")
            safe_click(driver, update_button, "'Update' button")
            time.sleep(10)

            # Click "Export Portfolio" link
            print("Looking for 'Export Portfolio' link...")
            export_link = wait_and_find_element(driver, By.ID, "downloadPortfolio", timeout=20)
            if not export_link:
                raise NoSuchElementException("Could not find 'Export Portfolio' link")
            safe_click(driver, export_link, "'Export Portfolio' link")

        # Handle cookie consent
        handle_cookie_consent(driver)

        # Login
        login(driver, "plessasdimitrios@yahoo.com", "QsDXJn8m@n@Li?3Y")
        
        # Wait for page to load after login
        print("Waiting for page to load after login...")
        time.sleep(20)  # Wait longer for the page to fully load
        
        # Handle portfolio buttons
        handle_portfolio_buttons(driver)
        
        # Wait for download to start
        print("Waiting for download...")
        time.sleep(5)
        
        # Process the downloaded portfolio
        print("Processing downloaded portfolio...")
        process_portfolio()
        
        return True
        
    except (NoSuchElementException, ElementClickInterceptedException) as e:
        print(f"Element interaction error: {str(e)}")
        return False
    except WebDriverException as e:
        print(f"WebDriver error: {str(e)}")
        return False
    except (OSError, IOError) as e:
        print(f"File system error: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if driver:
            print("Closing Chrome...")
            driver.quit()

if __name__ == "__main__":
    download_portfolio()