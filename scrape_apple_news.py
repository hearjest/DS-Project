"""
Apple Stock News Scraper
Scrapes news articles about Apple (AAPL) stock from Google News and extracts full article content.
Saves articles chronologically from oldest to newest.
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Try to import newspaper3k for article extraction (optional but recommended)
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("Warning: newspaper3k not installed. Install with: pip install newspaper3k")
    print("Will use basic content extraction instead.")

# Configuration
OUTPUT_CSV = "apple_news_articles.csv"
OUTPUT_JSON = "apple_news_articles.json"
SEARCH_TERMS = ["Apple stock", "AAPL", "Apple Inc", "Apple shares"]
MAX_ARTICLES_PER_QUERY = 100  # Google News RSS typically returns ~10-20 per query
DELAY_BETWEEN_REQUESTS = 1  # Seconds to wait between requests (be respectful)
REQUEST_TIMEOUT = 10  # Seconds before timing out

# NewsAPI configuration (optional - get free API key from https://newsapi.org/)
NEWSAPI_KEY = ""
USE_NEWSAPI = True  # Set to True if you have NewsAPI key


def get_google_news_rss(query, num_results=100, sort_by_date=True):
    """
    Get news articles from Google News RSS feed.
    
    Args:
        query: Search query string
        num_results: Maximum number of results (Google News RSS typically returns 10-20)
        sort_by_date: If True, sort by date (newest first in RSS, we'll reverse)
    
    Returns:
        List of article dictionaries
    """
    articles = []
    
    # Google News RSS URL
    # Note: Google News RSS is limited and may not return full results
    base_url = "https://news.google.com/rss"
    
    # Build search query
    search_url = f"{base_url}/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    
    if sort_by_date:
        search_url += "&sort=date"
    
    try:
        print(f"Fetching Google News RSS for: {query}")
        feed = feedparser.parse(search_url)
        
        if feed.bozo:
            print(f"  Warning: Feed parsing had issues: {feed.bozo_exception}")
        
        for entry in feed.entries[:num_results]:
            article = {
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'published_parsed': entry.get('published_parsed', None),
                'summary': entry.get('summary', ''),
                'source': entry.get('source', {}).get('title', '') if 'source' in entry else '',
                'query': query
            }
            
            # Parse date
            if article['published_parsed']:
                try:
                    article['published_date'] = datetime(*article['published_parsed'][:6])
                except:
                    article['published_date'] = None
            else:
                article['published_date'] = None
            
            articles.append(article)
            time.sleep(0.1)  # Small delay between entries
        
        print(f"  Found {len(articles)} articles")
        
    except Exception as e:
        print(f"  Error fetching Google News RSS: {e}")
    
    return articles


def get_newsapi_articles(query, from_date=None, to_date=None, sort_by='publishedAt'):
    """
    Get news articles from NewsAPI (requires API key).
    
    Args:
        query: Search query string
        from_date: Start date (YYYY-MM-DD) or None for all time
        to_date: End date (YYYY-MM-DD) or None for today
        sort_by: 'publishedAt' (newest first) or 'relevancy'
    
    Returns:
        List of article dictionaries
    """
    if not NEWSAPI_KEY:
        print("NewsAPI key not provided. Skipping NewsAPI.")
        return []
    
    articles = []
    base_url = "https://newsapi.org/v2/everything"
    
    params = {
        'q': query,
        'apiKey': NEWSAPI_KEY,
        'language': 'en',
        'sortBy': sort_by,
        'pageSize': 100,  # Max per request
        'page': 1
    }
    
    if from_date:
        params['from'] = from_date
    if to_date:
        params['to'] = to_date
    
    try:
        print(f"Fetching NewsAPI for: {query}")
        response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'ok':
            for article_data in data.get('articles', []):
                article = {
                    'title': article_data.get('title', ''),
                    'link': article_data.get('url', ''),
                    'published': article_data.get('publishedAt', ''),
                    'published_date': None,
                    'summary': article_data.get('description', ''),
                    'content': article_data.get('content', ''),  # NewsAPI sometimes has content
                    'source': article_data.get('source', {}).get('name', ''),
                    'query': query
                }
                
                # Parse date
                if article['published']:
                    try:
                        article['published_date'] = datetime.fromisoformat(
                            article['published'].replace('Z', '+00:00')
                        )
                    except:
                        try:
                            article['published_date'] = datetime.strptime(
                                article['published'], '%Y-%m-%dT%H:%M:%SZ'
                            )
                        except:
                            article['published_date'] = None
                
                articles.append(article)
            
            print(f"  Found {len(articles)} articles")
        else:
            print(f"  NewsAPI error: {data.get('message', 'Unknown error')}")
    
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching NewsAPI: {e}")
    except Exception as e:
        print(f"  Unexpected error with NewsAPI: {e}")
    
    return articles


def extract_article_content(url, title=""):
    """
    Extract full article content from a URL.
    
    Args:
        url: Article URL
        title: Article title (for fallback)
    
    Returns:
        Full article text or empty string
    """
    if not url:
        return ""
    
    # Try newspaper3k first (best results)
    if NEWSPAPER_AVAILABLE:
        try:
            article = Article(url)
            article.download()
            article.parse()
            content = article.text.strip()
            if content and len(content) > 100:  # Only return if substantial content
                return content
        except Exception as e:
            pass  # Fall back to other methods
    
    # Fallback: Try to extract from HTML
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content
        # Common article content selectors
        content_selectors = [
            'article',
            '[role="article"]',
            '.article-content',
            '.article-body',
            '.post-content',
            '.entry-content',
            'main',
            '.content',
            '#content'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text(strip=True) for elem in elements])
                if len(content) > 200:  # Substantial content found
                    break
        
        # If no specific content found, try to get all paragraphs
        if not content or len(content) < 200:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        # Clean up content
        content = ' '.join(content.split())  # Normalize whitespace
        
        if len(content) > 100:
            return content
    
    except Exception as e:
        pass
    
    return ""


def scrape_apple_news(use_newsapi=False, max_articles=None, date_range=None):
    """
    Main function to scrape Apple news articles.
    
    Args:
        use_newsapi: Whether to use NewsAPI (requires API key)
        max_articles: Maximum total articles to collect (None for all)
        date_range: Tuple of (start_date, end_date) as strings 'YYYY-MM-DD' or None
    
    Returns:
        DataFrame with all articles
    """
    all_articles = []
    seen_urls = set()  # Track URLs to avoid duplicates
    
    print("=" * 80)
    print("APPLE STOCK NEWS SCRAPER")
    print("=" * 80)
    print(f"Search terms: {', '.join(SEARCH_TERMS)}")
    print(f"Output files: {OUTPUT_CSV}, {OUTPUT_JSON}")
    print("=" * 80)
    
    # Collect articles from all sources
    for query in SEARCH_TERMS:
        if max_articles and len(all_articles) >= max_articles:
            break
        
        # Google News RSS
        google_articles = get_google_news_rss(query, num_results=MAX_ARTICLES_PER_QUERY)
        for article in google_articles:
            if article['link'] and article['link'] not in seen_urls:
                seen_urls.add(article['link'])
                all_articles.append(article)
        
        time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # NewsAPI (if enabled and key available)
        if use_newsapi and NEWSAPI_KEY:
            newsapi_articles = get_newsapi_articles(
                query,
                from_date=date_range[0] if date_range else None,
                to_date=date_range[1] if date_range else None
            )
            for article in newsapi_articles:
                if article['link'] and article['link'] not in seen_urls:
                    seen_urls.add(article['link'])
                    all_articles.append(article)
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        if max_articles and len(all_articles) >= max_articles:
            break
    
    print(f"\nTotal unique articles collected: {len(all_articles)}")
    
    # Extract full content for each article
    print("\nExtracting full article content...")
    print("(This may take a while - scraping each article URL)")
    
    for i, article in enumerate(all_articles):
        if (i + 1) % 10 == 0:
            print(f"  Processing article {i+1}/{len(all_articles)}...")
        
        # Only extract if we don't already have content
        if 'content' not in article or not article['content']:
            content = extract_article_content(article['link'], article['title'])
            article['content'] = content
            article['content_length'] = len(content)
        else:
            article['content_length'] = len(article.get('content', ''))
        
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_articles)
    
    # Sort by date (oldest first)
    if 'published_date' in df.columns:
        df = df.sort_values('published_date', na_position='last')
        df = df.reset_index(drop=True)
        
        # Format date as YYYY-MM-DD for matching with stock data (like 1995-10-17)
        df['date'] = df['published_date'].dt.strftime('%Y-%m-%d')
    
    # Select and order columns (date first for easy matching with stock data)
    columns_order = [
        'date', 'published_date', 'published', 'title', 'content', 'summary',
        'source', 'link', 'query', 'content_length'
    ]
    available_columns = [col for col in columns_order if col in df.columns]
    df = df[available_columns]
    
    # Format published_date column to YYYY-MM-DD for CSV output
    if 'published_date' in df.columns and df['published_date'].dtype != 'object':
        df['published_date'] = pd.to_datetime(df['published_date']).dt.strftime('%Y-%m-%d')
    
    # Save to CSV (dates will be in YYYY-MM-DD format)
    print(f"\nSaving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"  Saved {len(df)} articles to CSV")
    
    # Save to JSON (for easier programmatic access)
    print(f"Saving to {OUTPUT_JSON}...")
    df.to_json(OUTPUT_JSON, orient='records', date_format='iso', indent=2)
    print(f"  Saved {len(df)} articles to JSON")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SCRAPING SUMMARY")
    print("=" * 80)
    print(f"Total articles: {len(df)}")
    if 'published_date' in df.columns:
        date_range = df['published_date'].dropna()
        if len(date_range) > 0:
            print(f"Date range: {date_range.min()} to {date_range.max()}")
    if 'content_length' in df.columns:
        avg_length = df['content_length'].mean()
        print(f"Average article length: {avg_length:.0f} characters")
        print(f"Articles with content: {(df['content_length'] > 100).sum()}/{len(df)}")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    # Example usage
    
    # Option 1: Basic scraping (Google News RSS only, no API key needed)
    print("Starting basic scraping with Google News RSS...")
    df = scrape_apple_news(use_newsapi=False)
    
    # Option 2: With NewsAPI (requires API key)
    # Set environment variable: export NEWSAPI_KEY="your_key_here"
    # Or set it in the script above
    df = scrape_apple_news(use_newsapi=True, max_articles=500)
    
    # Option 3: Specific date range
    df = scrape_apple_news(
         use_newsapi=True,
         date_range=('1978-01-01', '2025-12-01')
     )
    
    print("\nScraping complete!")
    print(f"Results saved to: {OUTPUT_CSV} and {OUTPUT_JSON}")

