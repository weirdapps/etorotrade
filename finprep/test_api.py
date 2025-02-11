from support.load_env import load_environment
from support.get_data import (
    fetch_dcf_data,
    fetch_piotroski_score,
    fetch_price_target_data,
    fetch_analyst_recommendations,
    fetch_financial_score
)

def test_endpoints():
    api_key = load_environment()
    ticker = "AAPL"  # Using Apple as a test case
    
    print("\nTesting DCF endpoint:")
    dcf_data = fetch_dcf_data(ticker, api_key)
    print(f"DCF data: {dcf_data}")
    
    print("\nTesting Piotroski endpoint:")
    piotroski_data = fetch_piotroski_score(ticker, api_key)
    print(f"Piotroski data: {piotroski_data}")
    
    print("\nTesting Price Target endpoint:")
    price_target_data = fetch_price_target_data(ticker, api_key, "2023-01-01")
    print(f"Price target data: {price_target_data}")
    
    print("\nTesting Analyst Recommendations endpoint:")
    analyst_data = fetch_analyst_recommendations(ticker, api_key)
    print(f"Analyst data: {analyst_data}")
    
    print("\nTesting Financial Score endpoint:")
    score_data = fetch_financial_score(ticker, api_key)
    print(f"Financial score data: {score_data}")

if __name__ == "__main__":
    test_endpoints()