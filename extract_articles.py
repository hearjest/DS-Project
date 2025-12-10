#!/usr/bin/env python3
"""
Extract Alpha Vantage articles from Jupyter notebook output or recreate from CSV
"""
import json
import pandas as pd
import os

# Method 1: Check if CSV already exists (it does!)
csv_path = "C:/Users/brian/.vscode/dsproject/DS-Project/alpha_vantage_news_sentiment.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[OK] CSV file found with {len(df)} articles!")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"/nFirst 5 articles:")
    print(df[['time', 'title', 'source']].head())
    print(f"/n[OK] Your data is already saved in: {os.path.abspath(csv_path)}")
else:
    print("[ERROR] CSV not found. Trying to extract from notebook...")
    
    # Method 2: Try to extract from notebook output cells
    notebook_path = "s4test.ipynb"
    if os.path.exists(notebook_path):
        print(f"Reading notebook: {notebook_path}")
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        articles_found = []
        for cell in nb.get('cells', []):
            # Look for output that contains article data
            for output in cell.get('outputs', []):
                if 'text' in output:
                    text = ''.join(output['text'])
                    # Try to find JSON data in outputs
                    if 'unique_articles' in text or 'all_articles_data' in text:
                        print(f"Found potential article data in cell output")
<<<<<<< HEAD
         
=======
        
>>>>>>> 5da5bcf (update)
        if articles_found:
            print(f"Found {len(articles_found)} articles in notebook output")
        else:
            print("Could not extract from notebook output")
            print("/n[WARNING] The notebook output doesn't contain the raw article data.")
            print("   The data was likely only stored in variables during execution.")
            print("   You'll need to re-run the cell, but add this at the end to save immediately:")
            print("/n   # Emergency save - add this to your notebook cell:")
            print("   import json")
            print("   with open('alpha_vantage_articles_RAW.json', 'w') as f:")
            print("       json.dump(unique_articles, f, indent=2)")
            print("   print(f'Saved {len(unique_articles)} articles to JSON')")

# Method 3: Show summary stats
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("/n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    print(f"Total articles: {len(df)}")
    print(f"/nSentiment distribution:")
    print(df['overall_sentiment'].value_counts())
    print(f"/nAAPL-specific articles (relevance > 0):")
    aapl_articles = df[df['aapl_sentiment'] > 0]
    print(f"  Count: {len(aapl_articles)}")
    if len(aapl_articles) > 0:
        print(f"  AAPL sentiment distribution:")
        print(aapl_articles['aapl_label'].value_counts())

