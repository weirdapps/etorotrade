import yfinance as yf
from datetime import datetime
import textwrap
import html

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
        return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
    except:
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

def get_url(content):
    """Safely extract URL from content"""
    if not content:
        return 'N/A'
    
    # Try different possible URL locations
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

def format_news(news):
    print_section("LATEST NEWS")
    for item in news:
        try:
            content = item.get('content', {})
            if not content:
                continue
            
            title = content.get('title', 'N/A')
            print(f"\n{Colors.BOLD}• {title}{Colors.ENDC}")
            
            timestamp = format_timestamp(content.get('pubDate', ''))
            print(f"   {Colors.BLUE}Published:{Colors.ENDC} {timestamp}")
            
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                provider_name = provider.get('displayName', 'N/A')
            else:
                provider_name = 'N/A'
            print(f"   {Colors.BLUE}Publisher:{Colors.ENDC} {provider_name}")
            
            # Get summary from description or summary field
            summary = content.get('summary', content.get('description', ''))
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

def main():
    print(f"{Colors.BOLD}Yahoo Finance News{Colors.ENDC}")
    
    # Get and display news
    ticker = yf.Ticker("GOOGL")
    news = ticker.news
    format_news(news)

if __name__ == "__main__":
    main()
