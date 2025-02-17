from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
import pandas as pd

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

def login(driver, email, password):
    print("Attempting to log in...")
    time.sleep(10)  # Give more time for initial page load
    
    # Click on the sign-in button
    try:
        print("Looking for sign-in button...")
        sign_in_elements = driver.find_elements(By.CLASS_NAME, "action-button")
        if sign_in_elements:
            for element in sign_in_elements:
                if "Sign in" in element.text:
                    print(f"Found sign-in button with text: {element.text}")
                    time.sleep(2)
                    driver.execute_script("arguments[0].click();", element)
                    print("Clicked sign-in button")
                    break
    except Exception as e:
        print(f"Error with initial sign-in: {str(e)}")
        raise

    time.sleep(5)  # Wait longer for auth UI

    # Try to find and click the email sign-in button
    print("Looking for email sign-in button...")
    try:
        email_sign_in = wait_and_find_element(driver, By.XPATH, "//button[contains(., 'Sign in with email')]")
        if email_sign_in:
            time.sleep(2)
            driver.execute_script("arguments[0].click();", email_sign_in)
            print("Clicked email sign-in button")
        else:
            raise Exception("Email sign-in button not found")
    except Exception as e:
        print(f"Error with email sign-in: {str(e)}")
        raise

    time.sleep(5)

    # Try to find the email input field
    print("Looking for email input...")
    email_input = None
    try:
        email_input = wait_and_find_element(driver, By.ID, "ui-sign-in-email-input")
        if not email_input:
            email_input = wait_and_find_element(driver, By.XPATH, "//input[@type='email']")
    except Exception as e:
        print(f"Error finding email input: {str(e)}")
        raise

    if email_input:
        print("Entering email...")
        email_input.clear()
        time.sleep(1)
        email_input.send_keys(email)
        time.sleep(2)
        
        # Look for the Next button
        print("Looking for Next button...")
        try:
            next_button = wait_and_find_element(driver, By.XPATH, "//button[contains(text(), 'Next') or contains(text(), 'NEXT')]")
            if next_button:
                time.sleep(1)
                driver.execute_script("arguments[0].click();", next_button)
                print("Clicked Next button")
            else:
                print("No Next button found, trying to submit form...")
                email_input.send_keys(Keys.RETURN)
                print("Sent RETURN key to email input")
        except Exception as e:
            print(f"Error with Next button: {str(e)}")
            email_input.send_keys(Keys.RETURN)
            print("Sent RETURN key as fallback")

        time.sleep(5)  # Wait longer for password field

        # Look for password input
        print("Looking for password input...")
        max_attempts = 3
        password_input = None
        
        for attempt in range(max_attempts):
            try:
                # Try different methods to find password input
                try:
                    password_input = wait_and_find_element(driver, By.XPATH, "//input[@type='password']", timeout=5)
                except:
                    try:
                        password_input = wait_and_find_element(driver, By.CSS_SELECTOR, "input[type='password']", timeout=5)
                    except:
                        password_input = wait_and_find_element(driver, By.XPATH, "//input[contains(@class, 'password')]", timeout=5)
                
                if password_input:
                    print("Found password input, entering password...")
                    time.sleep(1)
                    password_input.clear()
                    password_input.send_keys(password)
                    time.sleep(2)
                    
                    # Try to find sign in button
                    sign_in_button = wait_and_find_element(driver, By.XPATH, "//button[contains(text(), 'Sign in') or contains(text(), 'SIGN IN')]", timeout=5)
                    if sign_in_button:
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", sign_in_button)
                        print("Clicked Sign in button")
                    else:
                        print("No Sign in button found, trying to submit form...")
                        password_input.send_keys(Keys.RETURN)
                        print("Sent RETURN key to password input")
                    
                    break  # Exit the loop if successful
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                    time.sleep(3)
                else:
                    print("All attempts failed")
                    raise
        
        if not password_input:
            raise Exception("Password input not found after all attempts")
        
        time.sleep(10)  # Wait longer for login to complete
    else:
        raise Exception("Email input not found")

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
        
        # Wait for and accept cookies if present
        try:
            print("Looking for cookie consent...")
            accept_button = wait_and_find_element(driver, By.XPATH, "//button[contains(text(), 'Accept All')]")
            if accept_button:
                time.sleep(1)
                driver.execute_script("arguments[0].click();", accept_button)
                print("Accepted cookies")
                time.sleep(2)
        except:
            print("No cookie consent needed or already accepted")

        # Login
        login(driver, "plessasdimitrios@yahoo.com", "QsDXJn8m@n@Li?3Y")
        
        # Wait for page to load after login
        print("Waiting for page to load after login...")
        time.sleep(20)  # Wait longer for the page to fully load
        
        # Click "Load this portfolio" button using its ID
        print("Looking for 'Load this portfolio' button...")
        load_button = wait_and_find_element(driver, By.ID, "loadPortfolioStats", timeout=20)
        if load_button:
            time.sleep(1)
            driver.execute_script("arguments[0].click();", load_button)
            print("Clicked 'Load this portfolio' button")
        else:
            raise Exception("Could not find 'Load this portfolio' button")
        time.sleep(10)
        
        # Click "Update" button using its ID
        print("Looking for 'Update' button...")
        update_button = wait_and_find_element(driver, By.ID, "updatePi", timeout=20)
        if update_button:
            time.sleep(1)
            driver.execute_script("arguments[0].click();", update_button)
            print("Clicked 'Update' button")
        else:
            raise Exception("Could not find 'Update' button")
        time.sleep(10)
        
        # Click "Export Portfolio" link using its ID
        print("Looking for 'Export Portfolio' link...")
        export_link = wait_and_find_element(driver, By.ID, "downloadPortfolio", timeout=20)
        if export_link:
            time.sleep(1)
            driver.execute_script("arguments[0].click();", export_link)
            print("Clicked 'Export Portfolio' link")
        else:
            raise Exception("Could not find 'Export Portfolio' link")
        
        # Wait for download to start
        print("Waiting for download...")
        time.sleep(5)
        
        # Process the downloaded portfolio
        print("Processing downloaded portfolio...")
        process_portfolio()
        
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if driver:
            print("Closing Chrome...")
            driver.quit()

if __name__ == "__main__":
    download_portfolio()