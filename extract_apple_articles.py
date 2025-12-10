import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths (relative to script directory)
nasdaq_file = os.path.join(SCRIPT_DIR, "STUFFALOTSOFSTUFF", "nasdaq_exteral_data.csv")
all_external_file = os.path.join(SCRIPT_DIR, "STUFFALOTSOFSTUFF", "All_external.csv")
output_file = os.path.join(SCRIPT_DIR, "apple_articles_compiled2.csv")
<<<<<<< HEAD
 
=======

>>>>>>> 5da5bcf (update)
# Word bank for Apple-related keywords
APPLE_KEYWORDS = [
    # Stock symbols and tickers
    'AAPL', 'aapl',
    # Company name variations
    'Apple', 'apple', 'APPLE',
    'Apple Inc', 'Apple Inc.', 'Apple Incorporated',
    # Common phrases
    'Apple stock', 'Apple shares', 'Apple equity',
    'Apple Financial', 'Apple finance', 'Apple earnings',
    'Apple revenue', 'Apple sales', 'Apple profit',
    'Apple CEO', 'Apple CFO', 'Apple executive',
    'Apple iPhone', 'Apple iPad', 'Apple Mac', 'Apple Watch',
    'Apple product', 'Apple services', 'Apple store',
    'Apple analyst', 'Apple rating', 'Apple target',
    'Apple dividend', 'Apple buyback', 'Apple split',
    # Common misspellings or variations
    'AAPL stock', 'AAPL shares',
    'Apple Computer',  # Historical name
]

# Create regex pattern for keyword matching (case-insensitive word boundaries)
keyword_pattern = '|'.join([re.escape(kw) for kw in APPLE_KEYWORDS])
keyword_regex = re.compile(r'\b(' + keyword_pattern + r')\b', re.IGNORECASE)

def contains_apple_keyword(text):
    """Check if text contains any Apple-related keyword"""
    if pd.isna(text) or text == '':
        return False
    text_str = str(text)
    return bool(keyword_regex.search(text_str))

def is_aapl_stock_symbol(symbol):
    """Check if stock symbol is AAPL (case-insensitive, handles variations)"""
    if pd.isna(symbol):
        return False
    symbol_str = str(symbol).upper().strip()
    # Check for exact match or contains AAPL
    return symbol_str == 'AAPL' or 'AAPL' in symbol_str

def extract_apple_articles_from_file(file_path, source_name, seen_urls, seen_titles_dates, output_file, write_buffer, buffer_size=50):
    """Extract Apple-related articles from a CSV file and write incrementally"""
    print(f"\n{'='*80}")
    print(f"Processing: {source_name}")
    print(f"File: {file_path}")
    print(f"{'='*80}")
    
    chunksize = 10000
    total_rows = 0
    chunk_num = 0
    articles_found = 0
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            chunk_num += 1
            total_rows += len(chunk)
            
            # Check which columns exist
            has_stock_symbol = 'Stock_symbol' in chunk.columns
            has_article_title = 'Article_title' in chunk.columns
            has_article = 'Article' in chunk.columns
            has_date = 'Date' in chunk.columns
            has_url = 'Url' in chunk.columns
            
            # Create a mask for Apple-related articles
            mask = pd.Series([False] * len(chunk), index=chunk.index)
            
            # Method 1: Check Stock_symbol column
            if has_stock_symbol:
                symbol_match = chunk['Stock_symbol'].apply(is_aapl_stock_symbol)
                mask = mask | symbol_match
            
            # Method 2: Check Article_title for keywords
            if has_article_title:
                title_match = chunk['Article_title'].apply(contains_apple_keyword)
                mask = mask | title_match
            
            # Method 3: Check Article content for keywords (if available)
            if has_article:
                article_match = chunk['Article'].apply(contains_apple_keyword)
                mask = mask | article_match
            
            # Method 4: Check other text columns if they exist
            text_columns = [col for col in chunk.columns if col not in 
                          ['Date', 'Stock_symbol', 'Url', 'Publisher', 'Author']]
            for col in text_columns:
                if chunk[col].dtype == 'object':  # String columns
                    col_match = chunk[col].apply(contains_apple_keyword)
                    mask = mask | col_match
            
            # Filter to Apple articles using boolean indexing
            apple_chunk = chunk[mask].copy()
            
            if len(apple_chunk) > 0:
                # Add source column to track which file it came from
                apple_chunk['Source_file'] = source_name
                
                # Filter out duplicates
                new_articles = []
                for idx, row in apple_chunk.iterrows():
                    # Check URL duplicate
                    is_duplicate = False
                    if has_url and pd.notna(row.get('Url')):
                        url_key = str(row['Url']).strip().lower()
                        if url_key in seen_urls:
                            is_duplicate = True
                        else:
                            seen_urls.add(url_key)
                    
                    # Check title+date duplicate
                    if not is_duplicate and has_article_title and has_date:
                        title = str(row.get('Article_title', '')).strip()
                        date = str(row.get('Date', '')).strip()
                        title_date_key = (title.lower(), date)
                        if title_date_key in seen_titles_dates:
                            is_duplicate = True
                        else:
                            seen_titles_dates.add(title_date_key)
                    
                    if not is_duplicate:
                        new_articles.append(row)
                
                if new_articles:
                    new_df = pd.DataFrame(new_articles)
                    write_buffer.append(new_df)
                    articles_found += len(new_df)
                    
                    # Write buffer to CSV when it reaches buffer_size (reduced from 50 to 10)
                    if len(write_buffer) >= 10:  # Changed from buffer_size
                        flush_buffer_to_csv(write_buffer, output_file)
                        write_buffer.clear()
                    
                    print(f"  Chunk {chunk_num}: Found {len(new_df)} new Apple articles (Total: {articles_found:,}, Rows processed: {total_rows:,})")
            
            # Progress update
            if chunk_num % 10 == 0:
                print(f"  Processed {chunk_num} chunks ({total_rows:,} rows, {articles_found:,} articles found)...")
        
        # FLUSH BUFFER AFTER EACH FILE (add this before the return statement)
        if write_buffer:
            print(f"  [INFO] Flushing {len(write_buffer)} batches after processing {source_name}...")
            flush_buffer_to_csv(write_buffer, output_file)
            write_buffer.clear()
        
        print(f"\n[OK] Completed: {total_rows:,} total rows processed, {articles_found:,} articles found")
        
    except Exception as e:
        print(f"  X Error processing {source_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always flush buffer even on error
        if write_buffer:
            print(f"  [INFO] Flushing buffer after error in {source_name}...")
            flush_buffer_to_csv(write_buffer, output_file)
            write_buffer.clear()
    
    return articles_found

def flush_buffer_to_csv(buffer, output_file):
    """Write accumulated articles from buffer to CSV file"""
    if not buffer:
        return
    
    combined = pd.concat(buffer, ignore_index=True)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
    
    # Append to CSV (mode='a' for append, header=False if file exists)
    combined.to_csv(output_file, mode='a', header=not file_exists, index=False)
    
    print(f"    [WRITTEN] Flushed {len(combined)} articles to CSV")

def main():
    print("="*80)
    print("APPLE ARTICLE EXTRACTION (INCREMENTAL WRITING)")
    print("="*80)
    print(f"\nKeyword bank contains {len(APPLE_KEYWORDS)} terms")
    print(f"Output file: {output_file}")
    
    # Initialize: Remove existing file or load existing data for duplicate tracking
    seen_urls = set()
    seen_titles_dates = set()
    write_buffer = []
    
    # If output file exists, load it to track duplicates
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"\n[INFO] Existing file found: {output_file}")
        try:
            existing_df = pd.read_csv(output_file, nrows=0)  # Just get headers
            print(f"  File exists with columns: {list(existing_df.columns)}")
            print("  Will append new articles and skip duplicates...")
            
            # Load existing URLs and titles to avoid duplicates
            print("  Loading existing articles for duplicate checking...")
            for chunk in pd.read_csv(output_file, chunksize=10000, low_memory=False):
                if 'Url' in chunk.columns:
                    seen_urls.update(chunk['Url'].dropna().astype(str).str.strip().str.lower())
                if 'Article_title' in chunk.columns and 'Date' in chunk.columns:
                    for _, row in chunk.iterrows():
                        title = str(row.get('Article_title', '')).strip()
                        date = str(row.get('Date', '')).strip()
                        if title and date:
                            seen_titles_dates.add((title.lower(), date))
            print(f"  Loaded {len(seen_urls)} existing URLs and {len(seen_titles_dates)} title+date pairs")
        except Exception as e:
            print(f"  [WARNING] Could not read existing file: {e}")
            print("  Starting fresh...")
            # Remove corrupted file
            if os.path.exists(output_file):
                os.remove(output_file)
    else:
        print(f"\n[INFO] Starting fresh - will create new file: {output_file}")
    
    total_articles_found = 0
    
    # Process nasdaq_external_data.csv
    try:
        articles = extract_apple_articles_from_file(
            nasdaq_file, "nasdaq_external_data", 
            seen_urls, seen_titles_dates, output_file, write_buffer
        )
        total_articles_found += articles
    except FileNotFoundError:
        print(f"\n[WARNING] File not found: {nasdaq_file}")
        print("  Skipping...")
    except Exception as e:
        print(f"\n[ERROR] Error processing {nasdaq_file}: {e}")
    
    # Process All_external.csv
    try:
        articles = extract_apple_articles_from_file(
            all_external_file, "All_external",
            seen_urls, seen_titles_dates, output_file, write_buffer
        )
        total_articles_found += articles
    except FileNotFoundError:
        print(f"\n[WARNING] File not found: {all_external_file}")
        print("  Skipping...")
    except Exception as e:
        print(f"\n[ERROR] Error processing {all_external_file}: {e}")
    
    # Flush any remaining articles in buffer
    if write_buffer:
        print(f"\n[INFO] Flushing final {len(write_buffer)} batches to CSV...")
        flush_buffer_to_csv(write_buffer, output_file)
        write_buffer.clear()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            # Load for statistics
            final_df = pd.read_csv(output_file, low_memory=False)
            print(f"Total articles (verified): {len(final_df):,}")
            
            print(f"\nColumns: {list(final_df.columns)}")
            
            if 'Source_file' in final_df.columns:
                print(f"\nBy source file:")
                print(final_df['Source_file'].value_counts())
            
            if 'Stock_symbol' in final_df.columns:
                print(f"\nStock_symbol values:")
                print(final_df['Stock_symbol'].value_counts().head(10))
            
            if 'Date' in final_df.columns:
                try:
                    final_df['Date'] = pd.to_datetime(final_df['Date'], errors='coerce')
                    valid_dates = final_df['Date'].dropna()
                    if len(valid_dates) > 0:
                        print(f"\nDate range: {valid_dates.min()} to {valid_dates.max()}")
                        date_counts = final_df.groupby(final_df['Date'].dt.to_period('M')).size()
                        print(f"\nArticles per month (first 10):")
                        print(date_counts.head(10))
                except:
                    print("  Could not parse dates")
        except Exception as e:
            print(f"[ERROR] Could not read final file: {e}")
    else:
        print("[WARNING] No output file created - no articles found!")
    
    print(f"\n{'='*80}")
    print("[OK] COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

