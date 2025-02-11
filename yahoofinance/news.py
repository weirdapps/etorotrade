import yfinance as yf
from datetime import datetime
import textwrap
import html
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from textblob import TextBlob

# Load environment variables
load_dotenv()
GOOGLE_NEWS_API_KEY = os.getenv('GOOGLE_NEWS_API_KEY')

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*50)
    print(f"{title}")
    print("="*50 + f"{Colors.ENDC}")

def format_timestamp(timestamp, is_google=False):
    try:
        if is_google:
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
        return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError) as e:
        return 'N/A'

def wrap_text(text, width=80, indent='   '):
    if not text:
        return text
    # Remove HTML tags and decode HTML entities
    text = html.unescape(text)
    # Simple HTML tag removal (for basic tags)
    while '<' in text and '>' in text:
        start = text.find('<')
        end = text.find('>', start)
        if end == -1:
            break
        text = text[:start] + text[end + 1:]
    wrapped_lines = textwrap.wrap(text, width=width)
    return f"\n{indent}".join(wrapped_lines)

def calculate_sentiment(title, summary):
    """
    Calculate sentiment score from -1 (most negative) to +1 (most positive)
    using both title and summary with title having more weight
    """
    # Title has 60% weight, summary has 40% weight
    title_weight = 0.6
    summary_weight = 0.4
    
    title_sentiment = TextBlob(title).sentiment.polarity
    summary_sentiment = TextBlob(summary or '').sentiment.polarity
    
    # Combine weighted sentiments
    combined_sentiment = (title_weight * title_sentiment + 
                        summary_weight * summary_sentiment)
    
    return combined_sentiment

def get_sentiment_color(sentiment):
    """Get color code based on sentiment value"""
    if sentiment < -0.2:
        return Colors.RED
    elif sentiment > 0.2:
        return Colors.GREEN
    return Colors.YELLOW

def get_google_news(ticker, limit=5):
    """Get news from Google News API with caching"""
    from .cache import news_cache
    
    # Create cache key
    cache_key = f"google_news_{ticker}_{limit}"
    print(f"\nChecking cache for {ticker} news...")
    
    # Try to get from cache first
    cached_news = news_cache.get(cache_key)
    if cached_news is not None:
        return cached_news
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': ticker,
            'apiKey': GOOGLE_NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': limit
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            print("Fetching fresh data from Google News API...")
            # Cache the results
            news_cache.set(cache_key, articles)
            return articles
        else:
            print(f"Error fetching Google News: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error accessing Google News API: {str(e)}")
        return []

def format_google_news(news, ticker):
    print_section(f"LATEST NEWS FOR {ticker}")
    for i, article in enumerate(news, 1):
        try:
            title = article.get('title', 'N/A')
            description = article.get('description', '')
            sentiment = calculate_sentiment(title, description)
            sentiment_color = get_sentiment_color(sentiment)
            
            print(f"\n{Colors.BOLD}• {title}{Colors.ENDC}")
            print(f"   {Colors.BLUE}Sentiment:{Colors.ENDC} "
                  f"{sentiment_color}{sentiment:.2f}{Colors.ENDC}")
            
            timestamp = format_timestamp(article.get('publishedAt', ''), is_google=True)
            print(f"   {Colors.BLUE}Published:{Colors.ENDC} {timestamp}")
            print(f"   {Colors.BLUE}Source:{Colors.ENDC} {article.get('source', {}).get('name', 'N/A')}")
            
            if article.get('description'):
                print(f"   {Colors.BLUE}Summary:{Colors.ENDC}")
                wrapped_summary = wrap_text(article['description'])
                print(f"   {wrapped_summary}")
            
            print(f"   {Colors.BLUE}Link:{Colors.ENDC} {Colors.YELLOW}{article.get('url', 'N/A')}{Colors.ENDC}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing news item: {str(e)}")
            continue

def get_url(content):
    """Safely extract URL from content"""
    if not content:
        return 'N/A'
    
    url_locations = [
        ('clickThroughUrl', 'url'),
        ('canonicalUrl', 'url'),
        ('link', None)
    ]
    
    for main_key, sub_key in url_locations:
        if main_key in content:
            if sub_key:
                if isinstance(content[main_key], dict):
                    return content[main_key].get(sub_key, 'N/A')
            else:
                return content[main_key]
    
    return 'N/A'

def format_yahoo_news(news, ticker, limit=5):
    print_section(f"LATEST NEWS FOR {ticker}")
    for i, item in enumerate(news[:limit], 1):
        try:
            content = item.get('content', {})
            if not content:
                continue
            
            title = content.get('title', 'N/A')
            summary = content.get('summary', content.get('description', ''))
            sentiment = calculate_sentiment(title, summary)
            sentiment_color = get_sentiment_color(sentiment)
            
            print(f"\n{Colors.BOLD}• {title}{Colors.ENDC}")
            print(f"   {Colors.BLUE}Sentiment:{Colors.ENDC} "
                  f"{sentiment_color}{sentiment:.2f}{Colors.ENDC}")
            
            timestamp = format_timestamp(content.get('pubDate', ''))
            print(f"   {Colors.BLUE}Published:{Colors.ENDC} {timestamp}")
            
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                provider_name = provider.get('displayName', 'N/A')
            else:
                provider_name = 'N/A'
            print(f"   {Colors.BLUE}Publisher:{Colors.ENDC} {provider_name}")
            
            if summary:
                print(f"   {Colors.BLUE}Summary:{Colors.ENDC}")
                wrapped_summary = wrap_text(summary)
                print(f"   {wrapped_summary}")
            
            url = get_url(content)
            print(f"   {Colors.BLUE}Link:{Colors.ENDC} {Colors.YELLOW}{url}{Colors.ENDC}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing news item: {str(e)}")
            continue

def get_portfolio_tickers():
    """Read tickers from portfolio.csv"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        portfolio_path = os.path.join(script_dir, 'input', 'portfolio.csv')
        df = pd.read_csv(portfolio_path)
        tickers = df['ticker'].tolist()
        return [ticker for ticker in tickers if not str(ticker).endswith('USD')]
    except Exception as e:
        print(f"Error reading portfolio file: {str(e)}")
        return []

def get_user_tickers():
    """Get tickers from user input"""
    tickers_input = input("Enter comma-separated tickers (e.g., AAPL,MSFT,GOOGL): ").strip()
    return [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

def get_news_source() -> str:
    """Get user's choice of news source."""
    print("\nSelect news source:")
    print("G - Google News API")
    print("Y - Yahoo Finance")
    
    while True:
        source = input("\nEnter your choice (G/Y): ").strip().upper()
        if source in ['G', 'Y']:
            return source
        print("Invalid choice. Please enter 'G' or 'Y'.")

def get_ticker_source() -> str:
    """Get user's choice of ticker input method."""
    print("\nSelect ticker input method:")
    print("P - Load tickers from portfolio.csv")
    print("I - Enter tickers manually")
    
    while True:
        choice = input("\nEnter your choice (P/I): ").strip().upper()
        if choice in ['P', 'I']:
            return choice
        print("Invalid choice. Please enter 'P' or 'I'.")

def fetch_yahoo_news(ticker: str) -> None:
    """Fetch and display news from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        from .cache import news_cache
        cache_key = f"yahoo_news_{ticker}"
        print(f"\nChecking cache for {ticker} news...")
        
        cached_news = news_cache.get(cache_key)
        if cached_news is not None:
            news = cached_news
        else:
            news = stock.news
            if news:
                print("Fetching fresh data from Yahoo Finance...")
                news_cache.set(cache_key, news)
        
        if news:
            format_yahoo_news(news, ticker, limit=5)
        else:
            print(f"\nNo news found for {ticker}")
    except Exception as e:
        print(f"\nError fetching news for {ticker}: {str(e)}")

def fetch_google_news(ticker: str) -> None:
    """Fetch and display news from Google News API."""
    try:
        news = get_google_news(ticker, limit=5)
        if news:
            format_google_news(news, ticker)
        else:
            print(f"\nNo news found for {ticker}")
    except Exception as e:
        print(f"\nError fetching news for {ticker}: {str(e)}")

def main():
    print(f"{Colors.BOLD}Stock Market News{Colors.ENDC}")
    
    source = get_news_source()
    choice = get_ticker_source()
    
    # Get tickers based on user choice
    tickers = get_portfolio_tickers() if choice == 'P' else get_user_tickers()
    
    if not tickers:
        print("No tickers found or provided.")
        return
        
    if choice == 'P':
        print(f"\nLoaded {len(tickers)} tickers from portfolio.csv")
    
    print(f"\nFetching news for: {', '.join(tickers)}")
    
    if source == 'G':
        if not GOOGLE_NEWS_API_KEY:
            print("Error: Google News API key not found in .env file")
            return
        for ticker in tickers:
            fetch_google_news(ticker)
    else:  # source == 'Y'
        for ticker in tickers:
            fetch_yahoo_news(ticker)

if __name__ == "__main__":
    main()