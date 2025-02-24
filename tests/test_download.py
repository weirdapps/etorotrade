import pytest
from unittest.mock import Mock, patch, mock_open, call
import os
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    WebDriverException
)
from selenium.webdriver.common.keys import Keys
from yahoofinance.download import (
    safe_click,
    setup_driver,
    wait_and_find_element,
    find_sign_in_button,
    handle_email_sign_in,
    handle_email_input,
    find_password_input,
    handle_password_submit,
    handle_password_input,
    login,
    process_portfolio,
    handle_cookie_consent,
    handle_portfolio_buttons,
    download_portfolio
)

@pytest.fixture
def mock_driver():
    return Mock()

@pytest.fixture
def mock_element():
    element = Mock()
    element.text = "Sign in"
    return element

@pytest.fixture
def mock_webdriver():
    with patch('selenium.webdriver.Chrome') as mock:
        yield mock

@pytest.fixture
def mock_dataframe():
    df = pd.DataFrame({
        'ticker': ['AAPL', 'BTC', 'ETH', 'XRP', 'SOL', '03690.HK', '01299.HK', '0700.HK', '9988.HK']
    })
    return df

def test_safe_click_success(mock_driver, mock_element):
    """Test successful element click"""
    result = safe_click(mock_driver, mock_element, "test element")
    assert result is True
    mock_driver.execute_script.assert_called_once_with("arguments[0].click();", mock_element)

def test_safe_click_intercepted(mock_driver, mock_element):
    """Test click intercepted error"""
    mock_driver.execute_script.side_effect = ElementClickInterceptedException("Click intercepted")
    with pytest.raises(ElementClickInterceptedException):
        safe_click(mock_driver, mock_element)

def test_safe_click_webdriver_error(mock_driver, mock_element):
    """Test WebDriver error during click"""
    mock_driver.execute_script.side_effect = WebDriverException("WebDriver error")
    with pytest.raises(WebDriverException):
        safe_click(mock_driver, mock_element)

def test_setup_driver(mock_webdriver):
    """Test Chrome driver setup with options"""
    setup_driver()
    mock_webdriver.assert_called_once()
    options = mock_webdriver.call_args[1]['options']
    assert '--no-sandbox' in options.arguments
    assert '--disable-dev-shm-usage' in options.arguments
    assert '--disable-web-security' in options.arguments
    assert '--disable-gpu' in options.arguments
    assert '--window-size=1200,800' in options.arguments

@patch('yahoofinance.download.WebDriverWait')
@patch('yahoofinance.download.EC')
def test_wait_and_find_element_visible(mock_ec, mock_webdriverwait, mock_driver):
    """Test waiting for visible element"""
    mock_element = Mock(name='found_element')
    mock_wait = Mock(name='wait')
    mock_wait.until.return_value = mock_element
    mock_webdriverwait.return_value = mock_wait
    mock_condition = Mock(name='condition')
    mock_ec.visibility_of_element_located.return_value = mock_condition
    
    element = wait_and_find_element(mock_driver, By.ID, "test-id", check_visibility=True)
    
    assert element == mock_element
    mock_webdriverwait.assert_called_once_with(mock_driver, 10)
    mock_ec.visibility_of_element_located.assert_called_once_with((By.ID, "test-id"))
    mock_wait.until.assert_called_once_with(mock_condition)

@patch('yahoofinance.download.WebDriverWait')
@patch('yahoofinance.download.EC')
def test_wait_and_find_element_present(mock_ec, mock_webdriverwait, mock_driver):
    """Test waiting for present element"""
    mock_element = Mock(name='found_element')
    mock_wait = Mock(name='wait')
    mock_wait.until.return_value = mock_element
    mock_webdriverwait.return_value = mock_wait
    mock_condition = Mock(name='condition')
    mock_ec.presence_of_element_located.return_value = mock_condition
    
    element = wait_and_find_element(mock_driver, By.ID, "test-id", check_visibility=False)
    
    assert element == mock_element
    mock_webdriverwait.assert_called_once_with(mock_driver, 10)
    mock_ec.presence_of_element_located.assert_called_once_with((By.ID, "test-id"))
    mock_wait.until.assert_called_once_with(mock_condition)

@patch('yahoofinance.download.WebDriverWait')
@patch('yahoofinance.download.EC')
def test_wait_and_find_element_timeout(mock_ec, mock_webdriverwait, mock_driver):
    """Test element wait timeout"""
    mock_wait = Mock(name='wait')
    mock_wait.until.side_effect = TimeoutException("Timeout")
    mock_webdriverwait.return_value = mock_wait
    mock_condition = Mock(name='condition')
    mock_ec.visibility_of_element_located.return_value = mock_condition
    
    element = wait_and_find_element(mock_driver, By.ID, "test-id")
    
    assert element is None
    mock_webdriverwait.assert_called_once_with(mock_driver, 10)
    mock_ec.visibility_of_element_located.assert_called_once_with((By.ID, "test-id"))
    mock_wait.until.assert_called_once_with(mock_condition)

def test_find_sign_in_button(mock_driver, mock_element):
    """Test finding sign in button"""
    mock_driver.find_elements.return_value = [mock_element]
    button = find_sign_in_button(mock_driver)
    assert button == mock_element
    mock_driver.find_elements.assert_called_once_with(By.CLASS_NAME, "action-button")

def test_find_sign_in_button_not_found(mock_driver):
    """Test sign in button not found"""
    mock_driver.find_elements.return_value = []
    button = find_sign_in_button(mock_driver)
    assert button is None

def test_find_sign_in_button_error(mock_driver):
    """Test error finding sign in button"""
    mock_driver.find_elements.side_effect = WebDriverException()
    with pytest.raises(WebDriverException):
        find_sign_in_button(mock_driver)

@patch('yahoofinance.download.wait_and_find_element')
@patch('yahoofinance.download.safe_click')
def test_handle_email_sign_in_success(mock_safe_click, mock_wait, mock_driver):
    """Test successful email sign in"""
    mock_element = Mock()
    mock_wait.return_value = mock_element
    
    handle_email_sign_in(mock_driver)
    mock_wait.assert_called_once_with(mock_driver, By.XPATH, "//button[contains(., 'Sign in with email')]")
    mock_safe_click.assert_called_once_with(mock_driver, mock_element, "email sign-in button")

@patch('yahoofinance.download.wait_and_find_element')
def test_handle_email_sign_in_not_found(mock_wait, mock_driver):
    """Test email sign in button not found"""
    mock_wait.return_value = None
    with pytest.raises(NoSuchElementException):
        handle_email_sign_in(mock_driver)

@patch('yahoofinance.download.wait_and_find_element')
def test_handle_email_input_success(mock_wait, mock_driver):
    """Test successful email input"""
    mock_input = Mock()
    mock_next = Mock()
    mock_wait.side_effect = [mock_input, mock_next]
    
    handle_email_input(mock_driver, "test@example.com")
    
    mock_input.clear.assert_called_once()
    mock_input.send_keys.assert_called_once_with("test@example.com")
    assert mock_wait.call_count == 2

@patch('yahoofinance.download.wait_and_find_element')
def test_handle_email_input_no_next_button(mock_wait, mock_driver):
    """Test email input with no next button"""
    mock_input = Mock()
    mock_wait.side_effect = [mock_input, None]
    
    handle_email_input(mock_driver, "test@example.com")
    
    mock_input.send_keys.assert_has_calls([
        call("test@example.com"),
        call(Keys.RETURN)
    ])

@patch('yahoofinance.download.wait_and_find_element')
def test_find_password_input_success(mock_wait, mock_driver):
    """Test finding password input"""
    mock_input = Mock()
    mock_wait.return_value = mock_input
    
    result = find_password_input(mock_driver)
    assert result == mock_input
    assert mock_wait.call_count == 1

@patch('yahoofinance.download.wait_and_find_element')
def test_find_password_input_not_found(mock_wait, mock_driver):
    """Test password input not found"""
    mock_wait.return_value = None
    result = find_password_input(mock_driver)
    assert result is None
    assert mock_wait.call_count == 3  # Tries all selectors

@patch('yahoofinance.download.safe_click')
def test_handle_password_submit_with_button(mock_safe_click, mock_driver):
    """Test password submission with button"""
    mock_input = Mock()
    mock_button = Mock()
    with patch('yahoofinance.download.wait_and_find_element', return_value=mock_button):
        handle_password_submit(mock_driver, mock_input, "password123")
        
    mock_input.clear.assert_called_once()
    mock_input.send_keys.assert_called_once_with("password123")
    mock_safe_click.assert_called_once_with(mock_driver, mock_button, "Sign in button")

@patch('yahoofinance.download.wait_and_find_element')
def test_handle_password_submit_no_button(mock_wait, mock_driver):
    """Test password submission without button"""
    mock_input = Mock()
    mock_wait.return_value = None
    
    handle_password_submit(mock_driver, mock_input, "password123")
    
    mock_input.send_keys.assert_has_calls([
        call("password123"),
        call(Keys.RETURN)
    ])

@patch('yahoofinance.download.find_password_input')
def test_handle_password_input_success(mock_find, mock_driver):
    """Test successful password input handling"""
    mock_input = Mock()
    mock_find.return_value = mock_input
    
    with patch('yahoofinance.download.handle_password_submit') as mock_submit:
        result = handle_password_input(mock_driver, "password123")
        
    assert result is True
    mock_submit.assert_called_once_with(mock_driver, mock_input, "password123")

@patch('yahoofinance.download.find_password_input')
def test_handle_password_input_retry(mock_find, mock_driver):
    """Test password input with retry"""
    mock_find.side_effect = [None, None, Mock()]
    
    with patch('yahoofinance.download.handle_password_submit'):
        result = handle_password_input(mock_driver, "password123")
        
    assert result is True
    assert mock_find.call_count == 3

@patch('yahoofinance.download.find_password_input')
def test_handle_password_input_failure(mock_find, mock_driver):
    """Test password input failure after retries"""
    mock_find.return_value = None
    
    with pytest.raises(NoSuchElementException):
        handle_password_input(mock_driver, "password123")
    
    assert mock_find.call_count == 3

def test_process_portfolio_no_downloads_dir():
    """Test portfolio processing with no downloads directory"""
    with patch('os.path.isdir', return_value=False):
        result = process_portfolio()
        assert result is None

def test_process_portfolio_no_csv_files():
    """Test portfolio processing with no CSV files"""
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]):
        result = process_portfolio()
        assert result is None

def test_fix_hk_ticker():
    """Test the fix_hk_ticker function for different HK tickers"""
    from yahoofinance.download import fix_hk_ticker
    
    # Test cases for fix_hk_ticker
    test_cases = [
        ('03690.HK', '3690.HK'),      # Leading zero, 5 digits -> remove zero
        ('01299.HK', '1299.HK'),      # Leading zero, 5 digits -> remove zero
        ('0700.HK', '0700.HK'),       # Leading zero, 4 digits -> unchanged
        ('9988.HK', '9988.HK'),       # No leading zero -> unchanged
        ('00700.HK', '700.HK'),       # Two leading zeros, 5 digits -> remove zeros
        ('AAPL', 'AAPL'),             # Non-HK ticker -> unchanged
        (None, None),                  # None value -> unchanged
        (123, 123)                     # Non-string -> unchanged
    ]
    
    for input_val, expected in test_cases:
        result = fix_hk_ticker(input_val)
        assert result == expected, f"Failed for {input_val}, got {result}, expected {expected}"

@patch('pandas.read_csv')
def test_process_portfolio_success(mock_read_csv, mock_dataframe):
    """Test successful portfolio processing"""
    mock_read_csv.return_value = mock_dataframe
    
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['portfolio.csv']), \
         patch('os.path.getctime', return_value=123456789), \
         patch('os.path.isfile', return_value=True), \
         patch('os.makedirs'), \
         patch('os.remove'), \
         patch('os.path.dirname', return_value='/test/path'), \
         patch('os.path.join', return_value='/test/path/output.csv'), \
         patch('pandas.DataFrame.to_csv'):
        process_portfolio()
        
        mock_read_csv.assert_called_once()
        mock_dataframe.to_csv.assert_called_once()
        
        # Verify crypto ticker updates
        assert 'BTC-USD' in mock_dataframe['ticker'].values
        assert 'ETH-USD' in mock_dataframe['ticker'].values
        assert 'XRP-USD' in mock_dataframe['ticker'].values
        assert 'SOL-USD' in mock_dataframe['ticker'].values
        
        # Verify HK ticker fix (5+ digits)
        assert '3690.HK' in mock_dataframe['ticker'].values  # 03690.HK -> 3690.HK
        assert '1299.HK' in mock_dataframe['ticker'].values  # 01299.HK -> 1299.HK
        assert '0700.HK' in mock_dataframe['ticker'].values  # 0700.HK unchanged (4 digits)
        assert '9988.HK' in mock_dataframe['ticker'].values  # 9988.HK unchanged

@patch('yahoofinance.download.wait_and_find_element')
@patch('yahoofinance.download.safe_click')
def test_handle_cookie_consent_present(mock_safe_click, mock_wait, mock_driver):
    """Test cookie consent handling when present"""
    mock_button = Mock()
    mock_wait.return_value = mock_button
    
    handle_cookie_consent(mock_driver)
    mock_safe_click.assert_called_once_with(mock_driver, mock_button, "cookie consent")

@patch('yahoofinance.download.wait_and_find_element')
def test_handle_cookie_consent_not_present(mock_wait, mock_driver):
    """Test cookie consent handling when not present"""
    mock_wait.return_value = None
    
    handle_cookie_consent(mock_driver)
    assert not mock_driver.execute_script.called

@patch('yahoofinance.download.wait_and_find_element')
@patch('yahoofinance.download.safe_click')
def test_handle_portfolio_buttons_success(mock_safe_click, mock_wait, mock_driver):
    """Test successful portfolio button handling"""
    mock_buttons = [Mock(), Mock(), Mock()]
    mock_wait.side_effect = mock_buttons
    
    handle_portfolio_buttons(mock_driver)
    
    assert mock_safe_click.call_count == 3
    assert mock_wait.call_count == 3

@patch('yahoofinance.download.wait_and_find_element')
def test_handle_portfolio_buttons_missing(mock_wait, mock_driver):
    """Test portfolio buttons not found"""
    mock_wait.return_value = None
    
    with pytest.raises(NoSuchElementException):
        handle_portfolio_buttons(mock_driver)

def test_download_portfolio_success():
    """Test successful portfolio download"""
    with patch('yahoofinance.download.setup_driver') as mock_setup, \
         patch('yahoofinance.download.handle_cookie_consent') as mock_cookie, \
         patch('yahoofinance.download.login') as mock_login, \
         patch('yahoofinance.download.handle_portfolio_buttons') as mock_buttons, \
         patch('yahoofinance.download.process_portfolio') as mock_process:
        
        result = download_portfolio()
        
        assert result is True
        mock_setup.assert_called_once()
        mock_cookie.assert_called_once()
        mock_login.assert_called_once()
        mock_buttons.assert_called_once()
        mock_process.assert_called_once()

def test_download_portfolio_element_error():
    """Test portfolio download with element error"""
    with patch('yahoofinance.download.setup_driver'), \
         patch('yahoofinance.download.handle_cookie_consent', side_effect=NoSuchElementException):
        
        result = download_portfolio()
        assert result is False

def test_download_portfolio_webdriver_error():
    """Test portfolio download with WebDriver error"""
    with patch('yahoofinance.download.setup_driver', side_effect=WebDriverException):
        result = download_portfolio()
        assert result is False

def test_download_portfolio_file_error():
    """Test portfolio download with file system error"""
    with patch('yahoofinance.download.setup_driver') as mock_setup, \
         patch('yahoofinance.download.handle_cookie_consent'), \
         patch('yahoofinance.download.login'), \
         patch('yahoofinance.download.handle_portfolio_buttons'), \
         patch('yahoofinance.download.process_portfolio', side_effect=IOError):
        
        result = download_portfolio()
        assert result is False
        mock_setup.return_value.quit.assert_called_once()