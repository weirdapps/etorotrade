import yfinance as yf

# Define the stock ticker
ticker_symbol = "MA"

# Get insider transactions data
msft = yf.Ticker(ticker_symbol)
insider_transactions = msft.get_insider_transactions()

# Check if data exists
if insider_transactions is not None and not insider_transactions.empty:
    # Print the full insider transactions dataset
    print("Insider Transactions Data:")
    print(insider_transactions)

    # Filter buy transactions
    buy_transactions = insider_transactions[insider_transactions["Transaction"] == "Buy"]["Value"].sum()
    total_transactions = insider_transactions["Value"].sum()

    # Calculate percentage of buys over total transactions
    buy_percentage = (buy_transactions / total_transactions) * 100 if total_transactions else 0

    print(f"\nPercentage of insider buys over total insider transactions for {ticker_symbol}: {buy_percentage:.2f}%")
else:
    print("No insider transaction data available for this ticker.")