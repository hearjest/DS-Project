"""
Naive Bayes + TF-IDF Sentiment Analysis for Apple Articles (BINARY CLASSIFICATION)
Based on methodology from EL_25_2_09.pdf paper

This script:
1. Reads articles from apple_articles_compiled2.csv (READ-ONLY)
2. Uses VADER scores as training labels (from apple_daily_sentiment.csv, READ-ONLY)
3. Trains MultinomialNB classifier with TF-IDF features (BINARY: positive vs negative)
4. Classifies all articles as positive/negative (combines neutral with negative)
5. Aggregates daily sentiment scores
6. Outputs to NEW file: apple_daily_sentiment_nb.csv (does NOT modify existing CSVs)
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from collections import defaultdict

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths (READ-ONLY input files)
ARTICLES_CSV = os.path.join(SCRIPT_DIR, 'apple_articles_compiled2.csv')
VADER_SENTIMENT_CSV = os.path.join(SCRIPT_DIR, 'apple_daily_sentiment.csv')

# Output file (NEW file, doesn't modify existing CSVs)
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'apple_daily_sentiment_nb.csv')

def preprocess_text(text):
    """
    Preprocess text for TF-IDF: lowercase, remove special chars, basic cleaning
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def load_articles():
    """
    Load articles from compiled CSV (READ-ONLY)
    """
    print(f"Loading articles from {ARTICLES_CSV}...")
    
    if not os.path.exists(ARTICLES_CSV):
        raise FileNotFoundError(f"Articles CSV not found: {ARTICLES_CSV}")
    
    # Read in chunks to handle large file
    chunks = []
    chunk_size = 50000
    
    for chunk in pd.read_csv(ARTICLES_CSV, chunksize=chunk_size, low_memory=False):
        # Combine title and article text
        chunk['combined_text'] = chunk.apply(
            lambda row: f"{str(row.get('Article_title', ''))} {str(row.get('Article', ''))}".strip(),
            axis=1
        )
        # Filter out empty articles
        chunk = chunk[chunk['combined_text'].str.len() > 10]
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} articles")
    
    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
    
    return df

def create_training_labels(df_articles, df_vader):
    """
    Create BINARY training labels using VADER scores as proxy
    Positive: VADER compound > 0.05
    Negative: VADER compound <= 0.05 (combines neutral and negative)
    """
    print("\nCreating BINARY training labels from VADER scores...")
    
    # Merge articles with VADER sentiment by date
    df_articles['date_only'] = df_articles['Date'].dt.date
    df_vader['date_only'] = pd.to_datetime(df_vader['Date']).dt.date
    
    df_merged = df_articles.merge(
        df_vader[['date_only', 'VADER_Compound_Mean']],
        on='date_only',
        how='left'
    )
    
    # Create BINARY labels based on VADER compound score
    def get_label(vader_score):
        if pd.isna(vader_score):
            return None
        if vader_score > 0.05:
            return 'positive'
        else:
            return 'negative'  # Combines neutral and negative
    
    df_merged['label'] = df_merged['VADER_Compound_Mean'].apply(get_label)
    
    # Filter out articles without labels
    df_labeled = df_merged[df_merged['label'].notna()].copy()
    
    print(f"Articles with labels: {len(df_labeled):,}")
    print(f"BINARY Label distribution:")
    print(df_labeled['label'].value_counts())
    
    return df_labeled

def train_naive_bayes_classifier(df_labeled):
    """
    Train MultinomialNB classifier with TF-IDF features
    """
    print("\n" + "="*80)
    print("TRAINING NAIVE BAYES CLASSIFIER WITH TF-IDF")
    print("="*80)
    
    # Preprocess text
    print("Preprocessing text...")
    texts = df_labeled['combined_text'].apply(preprocess_text).values
    labels = df_labeled['label'].values
    
    # Filter out empty texts
    valid_mask = [len(text) > 0 for text in texts]
    texts = np.array(texts)[valid_mask]
    labels = labels[valid_mask]
    
    print(f"Valid texts for training: {len(texts):,}")
    
    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining set: {len(X_train):,} articles")
    print(f"Test set: {len(X_test):,} articles")
    
    # Create TF-IDF vectorizer
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Top 10k features
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.95,  # Ignore words that appear in >95% of documents
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    # Train MultinomialNB classifier
    print("\nTraining MultinomialNB classifier...")
    clf = MultinomialNB(alpha=1.0)  # Laplace smoothing
    clf.fit(X_train_tfidf, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report (BINARY):")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix (BINARY):")
    cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
    print(f"Labels: ['positive', 'negative']")
    print(cm)
    print(f"\nTrue Positives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Negatives: {cm[1,1]}")
    
    return clf, vectorizer

def classify_all_articles(df_articles, clf, vectorizer):
    """
    Classify all articles using trained classifier
    """
    print("\n" + "="*80)
    print("CLASSIFYING ALL ARTICLES")
    print("="*80)
    
    # Preprocess all articles
    print("Preprocessing all articles...")
    texts = df_articles['combined_text'].apply(preprocess_text).values
    
    # Filter out empty texts
    valid_mask = np.array([len(text) > 0 for text in texts])
    valid_texts = np.array(texts)[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    print(f"Classifying {len(valid_texts):,} articles...")
    
    # Transform to TF-IDF
    X_tfidf = vectorizer.transform(valid_texts)
    
    # Predict
    predictions = clf.predict(X_tfidf)
    prediction_proba = clf.predict_proba(X_tfidf)
    
    # Create results dataframe
    df_results = df_articles.iloc[valid_indices].copy()
    df_results['NB_Sentiment'] = predictions
    
    # Add probabilities
    class_names = clf.classes_
    for i, class_name in enumerate(class_names):
        df_results[f'NB_{class_name}_Proba'] = prediction_proba[:, i]
    
    # For articles with empty text, assign 'negative' (default for binary)
    empty_indices = np.where(~valid_mask)[0]
    if len(empty_indices) > 0:
        df_empty = df_articles.iloc[empty_indices].copy()
        df_empty['NB_Sentiment'] = 'negative'
        for class_name in class_names:
            df_empty[f'NB_{class_name}_Proba'] = 1.0 / len(class_names)
        
        df_results = pd.concat([df_results, df_empty], ignore_index=True)
    
    return df_results

def aggregate_daily_sentiment(df_classified):
    """
    Aggregate sentiment scores by date
    """
    print("\n" + "="*80)
    print("AGGREGATING DAILY SENTIMENT")
    print("="*80)
    
    # Group by date
    df_classified['date_only'] = pd.to_datetime(df_classified['Date']).dt.date
    
    daily_data = []
    
    for date, group in df_classified.groupby('date_only'):
        # Count sentiment classes (BINARY: positive vs negative)
        sentiment_counts = group['NB_Sentiment'].value_counts()
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        total_count = len(group)
        
        # Calculate ratios
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        negative_ratio = negative_count / total_count if total_count > 0 else 0
        
        # Net sentiment (positive - negative)
        net_sentiment = positive_count - negative_count
        net_sentiment_ratio = positive_ratio - negative_ratio
        
        # Average probabilities
        proba_cols = [col for col in group.columns if col.startswith('NB_') and col.endswith('_Proba')]
        avg_probas = {}
        for col in proba_cols:
            avg_probas[col.replace('NB_', '').replace('_Proba', '')] = group[col].mean()
        
        daily_data.append({
            'Date': date,
            'NB_Article_Count': total_count,
            'NB_Positive_Count': positive_count,
            'NB_Negative_Count': negative_count,
            'NB_Positive_Ratio': positive_ratio,
            'NB_Negative_Ratio': negative_ratio,
            'NB_Net_Sentiment': net_sentiment,
            'NB_Net_Sentiment_Ratio': net_sentiment_ratio,
            **{f'NB_{k}_Mean_Proba': v for k, v in avg_probas.items()}
        })
    
    df_daily = pd.DataFrame(daily_data)
    df_daily = df_daily.sort_values('Date')
    
    print(f"Aggregated {len(df_daily)} days of sentiment data")
    print(f"Date range: {df_daily['Date'].min()} to {df_daily['Date'].max()}")
    print(f"\nSample daily statistics (BINARY):")
    print(df_daily[['Date', 'NB_Article_Count', 'NB_Positive_Count', 'NB_Negative_Count', 'NB_Positive_Ratio', 'NB_Net_Sentiment']].head(10))
    
    return df_daily

def main():
    """
    Main execution function
    """
    print("="*80)
    print("NAIVE BAYES + TF-IDF SENTIMENT ANALYSIS")
    print("="*80)
    print(f"Input files (READ-ONLY):")
    print(f"  - Articles: {ARTICLES_CSV}")
    print(f"  - VADER Sentiment: {VADER_SENTIMENT_CSV}")
    print(f"\nOutput file (NEW):")
    print(f"  - {OUTPUT_CSV}")
    print("="*80)
    
    # Step 1: Load articles
    df_articles = load_articles()
    
    # Step 2: Load VADER sentiment for training labels
    print(f"\nLoading VADER sentiment from {VADER_SENTIMENT_CSV}...")
    if not os.path.exists(VADER_SENTIMENT_CSV):
        raise FileNotFoundError(f"VADER sentiment CSV not found: {VADER_SENTIMENT_CSV}")
    
    df_vader = pd.read_csv(VADER_SENTIMENT_CSV)
    print(f"Loaded {len(df_vader):,} days of VADER sentiment data")
    
    # Step 3: Create training labels
    df_labeled = create_training_labels(df_articles, df_vader)
    
    # Step 4: Train classifier
    clf, vectorizer = train_naive_bayes_classifier(df_labeled)
    
    # Step 5: Classify all articles
    df_classified = classify_all_articles(df_articles, clf, vectorizer)
    
    # Step 6: Aggregate daily sentiment
    df_daily = aggregate_daily_sentiment(df_classified)
    
    # Step 7: Save to NEW CSV file (doesn't modify existing CSVs)
    print(f"\nSaving daily sentiment to {OUTPUT_CSV}...")
    df_daily.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved {len(df_daily)} days of sentiment data")
    print(f"✓ File: {OUTPUT_CSV}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Output saved to: {OUTPUT_CSV}")
    print("Note: No existing CSV files were modified.")

if __name__ == "__main__":
    main()

