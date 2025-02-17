import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from yahoofinance.download import setup_driver, wait_and_find_element, process_portfolio

@pytest.fixture
def mock_driver():
    return MagicMock()

@pytest.fixture
def mock_element():
    return MagicMock(spec=WebElement)

def test_setup_driver_options():
    """Test that Chrome options are set correctly"""
    with patch('yahoofinance.download.webdriver') as mock_webdriver:
        mock_options = MagicMock()
        mock_webdriver.ChromeOptions.return_value = mock_options
        
        setup_driver()
        
        # Verify all required options are set
        expected_arguments = [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-gpu',
            '--window-size=1200,800'
        ]
        
        for arg in expected_arguments:
            mock_options.add_argument.assert_any_call(arg)

def test_wait_and_find_element_visible(mock_driver, mock_element):
    """Test element finding with visibility check"""
    with patch('yahoofinance.download.WebDriverWait') as mock_wait:
        mock_wait.return_value.until.return_value = mock_element
        
        element = wait_and_find_element(mock_driver, By.ID, "test-id")
        
        assert element == mock_element
        mock_wait.assert_called_once()

def test_wait_and_find_element_not_found(mock_driver):
    """Test element not found scenario"""
    with patch('yahoofinance.download.WebDriverWait') as mock_wait:
        mock_wait.return_value.until.side_effect = Exception("Element not found")
        
        element = wait_and_find_element(mock_driver, By.ID, "test-id")
        
        assert element is None