import yfinance as yf
from datetime import datetime
import textwrap
import html
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def format_timestamp(timestamp):
    try:
        if not timestamp:
            return 'N/A'
        return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
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

def clean_html(text):
    """Remove HTML tags and normalize whitespace"""
    if not text:
        return ''
    
    # Remove HTML tags
    text = html.unescape(text)
    while '<' in text and '>' in text:
        start = text.find('<')
        end = text.find('>', start)
        if end == -1:
            break
        text = text[:start] + text[end + 1:]
    
    # Normalize whitespace and remove any remaining source references
    text = ' '.join(text.split())
    if '  ' in text:  # Double space often indicates source reference
        text = text.split('  ')[0]
    return text

def clean_text_for_display(text):
    """Clean text for display by removing source names and normalizing"""
    if not text:
        return ''
    
    # Remove source name if present after dash
    if ' - ' in text:
        text = text.split(' - ')[0]
    
    # Clean HTML and normalize
    text = clean_html(text)
    
    # Remove source name if it appears at the end
    words = text.split()
    for i in range(len(words)-1, -1, -1):
        if words[i] in ['TipRanks', 'CNBC', 'Reuters', 'Bloomberg', 'Yahoo', 'Finance', 'Nasdaq']:
            text = ' '.join(words[:i])
            break
    
    return text.strip()

def clean_text_for_sentiment(text):
    """Clean text for sentiment analysis by removing financial terms and normalizing"""
    if not text:
        return ''
    
    # First clean for display
    text = clean_text_for_display(text)
    
    # Then remove financial terms that might affect sentiment
    words = text.split()
    cleaned_words = []
    skip_next = False
    
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        
        # Skip if word is a stock symbol (all caps 1-5 letters)
        if word.isupper() and 1 <= len(word) <= 5 and word.isalpha():
            continue
        
        # Skip financial terms
        if word.lower() in ['q1', 'q2', 'q3', 'q4', 'fy', 'eps', 'revenue', 'earnings']:
            continue
        
        # Skip dollar amounts and following unit
        if word.startswith('$') or (word.startswith('(') and word[1:].startswith('$')):
            if i + 1 < len(words) and words[i+1].lower() in ['billion', 'million', 'trillion']:
                skip_next = True
            continue
        
        # Skip percentages
        if word.endswith('%') or (word.endswith(')') and word[:-1].endswith('%')):
            continue
        
        cleaned_words.append(word)
    
    return ' '.join(cleaned_words).strip()

def calculate_sentiment(title, summary):
    """
    Calculate sentiment score from -1 (most negative) to +1 (most positive)
    using both title and summary with title having more weight
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # Clean both title and summary
    clean_title = clean_text_for_sentiment(title)
    clean_summary = clean_text_for_sentiment(summary)
    
    # Title has 60% weight, summary has 40% weight
    title_weight = 0.6
    summary_weight = 0.4
    
    title_scores = analyzer.polarity_scores(clean_title)
    summary_scores = analyzer.polarity_scores(clean_summary)
    
    # Use compound scores which are already normalized between -1 and 1
    title_sentiment = title_scores['compound']
    summary_sentiment = summary_scores['compound']
    
    # Combine weighted sentiments
    combined_sentiment = (title_weight * title_sentiment +
                        summary_weight * summary_sentiment)
    
    return combined_sentiment

def get_sentiment_color(sentiment):
    """Get color code based on sentiment value"""
    if sentiment < -0.05:  # VADER's threshold for negative
        return Colors.RED
    elif sentiment > 0.05:  # VADER's threshold for positive
        return Colors.GREEN
    return Colors.YELLOW


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


def main():
    print(f"{Colors.BOLD}Stock Market News{Colors.ENDC}")
    
    choice = get_ticker_source()
    
    # Get tickers based on user choice
    tickers = get_portfolio_tickers() if choice == 'P' else get_user_tickers()
    
    if not tickers:
        print("No tickers found or provided.")
        return
        
    if choice == 'P':
        print(f"\nLoaded {len(tickers)} tickers from portfolio.csv")
    
    print(f"\nFetching news for: {', '.join(tickers)}")
    
    for ticker in tickers:
        fetch_yahoo_news(ticker)

if __name__ == "__main__":
    main()