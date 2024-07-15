from support.load_env import load_environment
from support.get_data import (
    fetch_earliest_valid_date
)
from support.man_data import extract_financial_metrics
from support.display import display_table

def main():
    try:
        api_key = load_environment()
        ticker = input("Enter the ticker symbol: ")
        
        # Fetch the earliest valid date
        start_date_data = fetch_earliest_valid_date(ticker, api_key)
        start_date = None
        
        # Ensure start_date_data is processed correctly
        if start_date_data:
            if isinstance(start_date_data, list):
                for record in start_date_data:
                    if record.get('eps') is not None or record.get('revenue') is not None:
                        start_date = record.get('date')
                        if isinstance(start_date, list):
                            start_date = start_date[0]  # Take the first element if it's a list
                        break
            else:
                print(f"Unexpected format for start_date_data: {start_date_data}")

        # Ensure start_date is a string before further processing
        if not isinstance(start_date, str):
            raise ValueError(f"Invalid start_date: {start_date}")
        
        financial_metrics = extract_financial_metrics(ticker, api_key, start_date)
        display_table([financial_metrics])

        print(f"Last earnings release date: {start_date}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()