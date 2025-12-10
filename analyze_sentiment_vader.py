import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import defaultdict

# VADER Sentiment Analysis
try: 
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("[ERROR] vaderSentiment not installed. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

# Comprehensive financial word bank for stock price prediction
POSITIVE_STOCK_WORDS = [
    # Earnings & Growth
    'beat', 'beats', 'beaten', 'exceed', 'exceeds', 'exceeded', 'surpass', 'surpasses', 'surpassed',
    'growth', 'growing', 'grew', 'expand', 'expands', 'expanded', 'expansion', 'expanding',
    'profit', 'profits', 'profitable', 'profitability', 'earnings', 'revenue', 'revenues',
    'gain', 'gains', 'gained', 'gainful', 'upside', 'upside potential',
    'increase', 'increases', 'increased', 'increasing', 'rise', 'rises', 'rose', 'rising',
    'surge', 'surges', 'surged', 'surging', 'soar', 'soars', 'soared', 'soaring',
    'rally', 'rallies', 'rallied', 'rallying', 'jump', 'jumps', 'jumped', 'jumping',
    
    # Positive Business Terms
    'strong', 'stronger', 'strongest', 'strength', 'strengths',
    'success', 'successful', 'succeed', 'succeeds', 'succeeded', 'succeeding',
    'outperform', 'outperforms', 'outperformed', 'outperforming', 'outperformance',
    'bullish', 'bull', 'bulls', 'bullishness',
    'buy', 'buys', 'buying', 'buyer', 'buyers', 'purchase', 'purchases', 'purchasing',
    'upgrade', 'upgrades', 'upgraded', 'upgrading', 'upward', 'upwards',
    'positive', 'positively', 'positivity',
    'optimistic', 'optimism', 'optimistically',
    'favorable', 'favorably', 'favor',
    'promising', 'promise', 'promises', 'promised',
    'momentum', 'momentous',
    
    # Product & Innovation
    'launch', 'launches', 'launched', 'launching', 'release', 'releases', 'released', 'releasing',
    'breakthrough', 'breakthroughs', 'innovation', 'innovations', 'innovative', 'innovate',
    'new product', 'new products', 'new service', 'new services',
    'premium', 'premiums', 'quality', 'qualities', 'high-quality',
    
    # Market Position
    'leader', 'leaders', 'leadership', 'leading', 'lead', 'leads', 'led',
    'dominant', 'dominate', 'dominates', 'dominated', 'dominating', 'dominance',
    'market share', 'market leader', 'top performer', 'best-in-class',
    'competitive', 'competitiveness', 'competitive advantage',
    
    # Financial Health
    'cash flow', 'cash flows', 'liquidity', 'liquid', 'solvent', 'solvency',
    'dividend', 'dividends', 'dividend yield', 'dividend increase',
    'buyback', 'buybacks', 'share buyback', 'stock buyback', 'repurchase', 'repurchases',
    'debt reduction', 'debt free', 'low debt', 'strong balance sheet',
    'margin expansion', 'margin improvement', 'improving margins',
    
    # Analyst & Rating Terms
    'outperform', 'strong buy', 'buy rating', 'upgrade to buy',
    'price target raised', 'price target increase', 'higher target',
    'analyst upgrade', 'upgraded rating', 'positive rating',
    'overweight', 'overweight rating',
    
    # Market Sentiment
    'rally', 'rallies', 'momentum', 'bull market', 'bull run',
    'breakout', 'breakouts', 'breakout above', 'new high', 'new highs', 'all-time high',
    'support', 'support level', 'strong support',
    'resistance broken', 'break resistance',
    
    # Company Performance
    'record', 'records', 'record-breaking', 'record high', 'record revenue',
    'milestone', 'milestones', 'achievement', 'achievements', 'achieve', 'achieved',
    'excellence', 'excellent', 'excellently',
    'efficient', 'efficiency', 'efficiencies',
    
    # Partnership & Deals
    'partnership', 'partnerships', 'partner', 'partners', 'partnered',
    'deal', 'deals', 'deal signed', 'contract', 'contracts', 'signed contract',
    'acquisition', 'acquisitions', 'acquire', 'acquires', 'acquired', 'acquiring',
    'merger', 'mergers', 'merge', 'merges', 'merged', 'merging',
    
    # Guidance & Outlook
    'guidance raised', 'guidance increase', 'raised guidance', 'positive guidance',
    'outlook positive', 'outlook improved', 'improved outlook', 'bright outlook',
    'forecast', 'forecasts', 'forecasted', 'forecasting', 'upward revision',
    
    # Recovery & Turnaround
    'recovery', 'recoveries', 'recover', 'recovers', 'recovered', 'recovering',
    'rebound', 'rebounds', 'rebounded', 'rebounding', 'bounce', 'bounces', 'bounced', 'bouncing',
    'turnaround', 'turnarounds', 'turn around', 'turned around',
    'revival', 'revive', 'revives', 'revived', 'reviving',
    
    # Technical Terms (Positive)
    'oversold bounce', 'support held', 'bullish pattern', 'uptrend', 'uptrending',
    'momentum building', 'accumulation', 'institutional buying',
]

NEGATIVE_STOCK_WORDS = [
    # Earnings & Losses
    'miss', 'misses', 'missed', 'missing', 'missed expectations', 'missed estimates',
    'disappoint', 'disappoints', 'disappointed', 'disappointing', 'disappointment', 'disappointments',
    'decline', 'declines', 'declined', 'declining', 'decrease', 'decreases', 'decreased', 'decreasing',
    'fall', 'falls', 'fell', 'falling', 'drop', 'drops', 'dropped', 'dropping',
    'plunge', 'plunges', 'plunged', 'plunging', 'plummet', 'plummets', 'plummeted', 'plummeting',
    'crash', 'crashes', 'crashed', 'crashing', 'collapse', 'collapses', 'collapsed', 'collapsing',
    'loss', 'losses', 'losing', 'lost', 'unprofitable', 'loss-making',
    'revenue decline', 'revenue drop', 'revenue fall', 'sales decline', 'sales drop',
    'earnings miss', 'earnings disappointment', 'weak earnings', 'poor earnings',
    
    # Negative Business Terms
    'weak', 'weaker', 'weakest', 'weakness', 'weaknesses', 'weaken', 'weakens', 'weakened', 'weakening',
    'failure', 'failures', 'fail', 'fails', 'failed', 'failing',
    'struggle', 'struggles', 'struggled', 'struggling', 'struggling company',
    'bearish', 'bear', 'bears', 'bearishness',
    'sell', 'sells', 'selling', 'seller', 'sellers', 'sale', 'sales',
    'downgrade', 'downgrades', 'downgraded', 'downgrading', 'downward', 'downwards',
    'negative', 'negatively', 'negativity',
    'pessimistic', 'pessimism', 'pessimistically',
    'unfavorable', 'unfavorably', 'disfavor',
    'concerning', 'concern', 'concerns', 'concerned',
    'worry', 'worries', 'worried', 'worrying', 'worrisome',
    'risk', 'risks', 'risky', 'risking', 'risked',
    'uncertain', 'uncertainty', 'uncertainties', 'uncertainly',
    
    # Market Position
    'lag', 'lags', 'lagged', 'lagging', 'laggard', 'laggards',
    'lose market share', 'market share loss', 'declining market share',
    'competition', 'competitive pressure', 'competitive threat', 'losing ground',
    'disrupt', 'disrupts', 'disrupted', 'disrupting', 'disruption', 'disruptions',
    
    # Financial Problems
    'debt', 'debts', 'indebted', 'debt burden', 'high debt', 'debt crisis',
    'bankruptcy', 'bankruptcies', 'bankrupt', 'insolvent', 'insolvency',
    'liquidity crisis', 'cash flow problem', 'cash flow negative',
    'dividend cut', 'dividend reduction', 'suspended dividend', 'no dividend',
    'credit downgrade', 'rating downgrade', 'downgraded credit',
    'margin compression', 'margin pressure', 'declining margins', 'margin squeeze',
    
    # Analyst & Rating Terms
    'underperform', 'strong sell', 'sell rating', 'downgrade to sell',
    'price target cut', 'price target reduction', 'lower target', 'target lowered',
    'analyst downgrade', 'downgraded rating', 'negative rating',
    'underweight', 'underweight rating',
    
    # Market Sentiment
    'selloff', 'selloffs', 'sell-off', 'sell-offs', 'rout', 'routs',
    'bear market', 'bear run', 'correction', 'corrections', 'market correction',
    'breakdown', 'breakdowns', 'break down', 'broke down',
    'new low', 'new lows', 'all-time low', '52-week low',
    'resistance', 'resistance level', 'hit resistance', 'rejected at resistance',
    'support broken', 'break support', 'broke support',
    
    # Company Problems
    'scandal', 'scandals', 'controversy', 'controversies', 'controversial',
    'lawsuit', 'lawsuits', 'legal action', 'legal trouble', 'litigation',
    'investigation', 'investigations', 'investigate', 'investigates', 'investigated', 'investigating',
    'regulatory', 'regulatory action', 'regulatory scrutiny', 'regulatory issue',
    'fine', 'fines', 'penalty', 'penalties', 'penalized',
    'recall', 'recalls', 'recalled', 'recalling',
    'defect', 'defects', 'defective', 'malfunction', 'malfunctions',
    'layoff', 'layoffs', 'lay off', 'laid off', 'job cuts', 'workforce reduction',
    'restructuring', 'restructure', 'restructures', 'restructured', 'restructuring charge',
    
    # Guidance & Outlook
    'guidance cut', 'guidance reduction', 'lowered guidance', 'negative guidance',
    'outlook negative', 'outlook worsened', 'worsened outlook', 'gloomy outlook',
    'forecast cut', 'forecast reduction', 'downward revision', 'revised down',
    'warning', 'warnings', 'warn', 'warns', 'warned', 'warning signs',
    
    # Decline & Downturn
    'downturn', 'downturns', 'down cycle', 'downcycle',
    'recession', 'recessions', 'recessionary', 'economic downturn',
    'slowdown', 'slowdowns', 'slow down', 'slowed down', 'slowing',
    'stagnation', 'stagnate', 'stagnates', 'stagnated', 'stagnating',
    'contraction', 'contractions', 'contract', 'contracts', 'contracted', 'contracting',
    
    # Technical Terms (Negative)
    'oversold', 'overbought', 'bearish pattern', 'downtrend', 'downtrending',
    'momentum loss', 'distribution', 'institutional selling',
    'death cross', 'bearish crossover',
    
    # Volatility & Risk
    'volatile', 'volatility', 'high volatility', 'extreme volatility',
    'uncertainty', 'uncertainties', 'uncertain', 'unpredictable',
    'risk', 'risks', 'risky', 'high risk', 'risk factors',
    
    # Competition & Threats
    'competition', 'competitive threat', 'losing to competitors', 'market share loss',
    'disruption', 'disrupted', 'disruptive technology', 'threatened by',
    'obsolete', 'obsolescence', 'outdated', 'outmoded',
]

def analyze_sentiment_vader(text, analyzer):
    """Analyze sentiment using VADER"""
    if pd.isna(text) or text == '':
        return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    text_str = str(text)
    scores = analyzer.polarity_scores(text_str)
    return scores

def get_word_bank_sentiment(text, positive_words, negative_words):
    """Calculate sentiment based on word bank presence"""
    if pd.isna(text) or text == '':
        return {'positive_words': 0, 'negative_words': 0, 'net_sentiment': 0}
    
    text_lower = str(text).lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    net_sentiment = pos_count - neg_count
    
    return {
        'positive_words': pos_count,
        'negative_words': neg_count,
        'net_sentiment': net_sentiment
    }

def process_articles_csv(csv_path, chunksize=50000):
    """Process articles CSV and aggregate sentiment by date"""
    if not VADER_AVAILABLE:
        print("[ERROR] VADER not available. Cannot proceed.")
        return None
    
    print("=" * 80)
    print("VADER SENTIMENT ANALYSIS")
    print("=" * 80)
    print(f"File: {csv_path}")
    print(f"Chunk size: {chunksize:,}")
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Date-level aggregation
    daily_sentiment = defaultdict(lambda: {
        'vader_compound': [],
        'vader_pos': [],
        'vader_neu': [],
        'vader_neg': [],
        'wordbank_pos': [],
        'wordbank_neg': [],
        'wordbank_net': [],
        'article_count': 0,
        'has_title': 0,
        'has_article': 0
    })
    
    chunk_num = 0
    total_rows = 0
    
    print("\nProcessing articles...")
    
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        
        # Parse dates
        chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce', utc=True)
        
        # Process each row
        for idx, row in chunk.iterrows():
            date = row.get('Date')
            if pd.isna(date):
                continue
            
            # Normalize date to date only (remove time)
            date_key = date.date() if hasattr(date, 'date') else pd.Timestamp(date).date()
            
            # Analyze Article_title
            title = row.get('Article_title', '')
            if pd.notna(title) and str(title).strip():
                daily_sentiment[date_key]['has_title'] += 1
                
                # VADER sentiment
                title_vader = analyze_sentiment_vader(title, analyzer)
                daily_sentiment[date_key]['vader_compound'].append(title_vader['compound'])
                daily_sentiment[date_key]['vader_pos'].append(title_vader['pos'])
                daily_sentiment[date_key]['vader_neu'].append(title_vader['neu'])
                daily_sentiment[date_key]['vader_neg'].append(title_vader['neg'])
                
                # Word bank sentiment
                title_words = get_word_bank_sentiment(title, POSITIVE_STOCK_WORDS, NEGATIVE_STOCK_WORDS)
                daily_sentiment[date_key]['wordbank_pos'].append(title_words['positive_words'])
                daily_sentiment[date_key]['wordbank_neg'].append(title_words['negative_words'])
                daily_sentiment[date_key]['wordbank_net'].append(title_words['net_sentiment'])
            
            # Analyze Article content
            article = row.get('Article', '')
            if pd.notna(article) and str(article).strip():
                daily_sentiment[date_key]['has_article'] += 1
                
                # VADER sentiment
                article_vader = analyze_sentiment_vader(article, analyzer)
                daily_sentiment[date_key]['vader_compound'].append(article_vader['compound'])
                daily_sentiment[date_key]['vader_pos'].append(article_vader['pos'])
                daily_sentiment[date_key]['vader_neu'].append(article_vader['neu'])
                daily_sentiment[date_key]['vader_neg'].append(article_vader['neg'])
                
                # Word bank sentiment
                article_words = get_word_bank_sentiment(article, POSITIVE_STOCK_WORDS, NEGATIVE_STOCK_WORDS)
                daily_sentiment[date_key]['wordbank_pos'].append(article_words['positive_words'])
                daily_sentiment[date_key]['wordbank_neg'].append(article_words['negative_words'])
                daily_sentiment[date_key]['wordbank_net'].append(article_words['net_sentiment'])
            
            daily_sentiment[date_key]['article_count'] += 1
        
        if chunk_num % 10 == 0:
            print(f"  Processed {chunk_num} chunks ({total_rows:,} rows)...")
    
    print(f"\n[OK] Processed {total_rows:,} rows across {chunk_num} chunks")
    print(f"Found {len(daily_sentiment)} unique dates")
    
    # Aggregate daily sentiment
    daily_df_data = []
    for date, data in sorted(daily_sentiment.items()):
        daily_df_data.append({
            'Date': pd.Timestamp(date),
            'Article_Count': data['article_count'],
            'Has_Title': data['has_title'],
            'Has_Article': data['has_article'],
            'VADER_Compound_Mean': np.mean(data['vader_compound']) if data['vader_compound'] else 0,
            'VADER_Compound_Std': np.std(data['vader_compound']) if data['vader_compound'] else 0,
            'VADER_Positive_Mean': np.mean(data['vader_pos']) if data['vader_pos'] else 0,
            'VADER_Neutral_Mean': np.mean(data['vader_neu']) if data['vader_neu'] else 0,
            'VADER_Negative_Mean': np.mean(data['vader_neg']) if data['vader_neg'] else 0,
            'WordBank_Positive_Sum': sum(data['wordbank_pos']) if data['wordbank_pos'] else 0,
            'WordBank_Negative_Sum': sum(data['wordbank_neg']) if data['wordbank_neg'] else 0,
            'WordBank_Net_Sentiment': sum(data['wordbank_net']) if data['wordbank_net'] else 0,
        })
    
    daily_df = pd.DataFrame(daily_df_data)
    daily_df = daily_df.sort_values('Date')
    
    return daily_df

def plot_sentiment_analysis(daily_df, output_dir='plots'):
    """Create comprehensive sentiment visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Apple Stock News Sentiment Analysis Over Time', fontsize=16, fontweight='bold')
    
    dates = daily_df['Date']
    
    # Plot 1: VADER Compound Sentiment
    ax1 = axes[0, 0]
    ax1.plot(dates, daily_df['VADER_Compound_Mean'], label='Mean Compound', linewidth=2, color='blue')
    ax1.fill_between(dates, 
                     daily_df['VADER_Compound_Mean'] - daily_df['VADER_Compound_Std'],
                     daily_df['VADER_Compound_Mean'] + daily_df['VADER_Compound_Std'],
                     alpha=0.2, color='blue', label='±1 Std Dev')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('VADER Compound Sentiment Score', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Compound Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: VADER Positive/Negative Breakdown
    ax2 = axes[0, 1]
    ax2.plot(dates, daily_df['VADER_Positive_Mean'], label='Positive', linewidth=2, color='green')
    ax2.plot(dates, daily_df['VADER_Negative_Mean'], label='Negative', linewidth=2, color='red')
    ax2.plot(dates, daily_df['VADER_Neutral_Mean'], label='Neutral', linewidth=2, color='gray', alpha=0.5)
    ax2.set_title('VADER Sentiment Breakdown', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sentiment Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Word Bank Net Sentiment
    ax3 = axes[1, 0]
    ax3.plot(dates, daily_df['WordBank_Net_Sentiment'], linewidth=2, color='purple')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_title('Word Bank Net Sentiment (Positive - Negative Words)', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Net Sentiment Count')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Word Bank Positive vs Negative
    ax4 = axes[1, 1]
    ax4.plot(dates, daily_df['WordBank_Positive_Sum'], label='Positive Words', linewidth=2, color='green')
    ax4.plot(dates, daily_df['WordBank_Negative_Sum'], label='Negative Words', linewidth=2, color='red')
    ax4.set_title('Word Bank Positive vs Negative Word Counts', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Word Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Article Count Over Time
    ax5 = axes[2, 0]
    ax5.bar(dates, daily_df['Article_Count'], alpha=0.6, color='steelblue', width=1)
    ax5.set_title('Number of Articles Per Day', fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Article Count')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Combined Sentiment Comparison
    ax6 = axes[2, 1]
    # Normalize both for comparison
    vader_norm = (daily_df['VADER_Compound_Mean'] - daily_df['VADER_Compound_Mean'].min()) / \
                 (daily_df['VADER_Compound_Mean'].max() - daily_df['VADER_Compound_Mean'].min() + 1e-10)
    wordbank_norm = (daily_df['WordBank_Net_Sentiment'] - daily_df['WordBank_Net_Sentiment'].min()) / \
                    (daily_df['WordBank_Net_Sentiment'].max() - daily_df['WordBank_Net_Sentiment'].min() + 1e-10)
    ax6.plot(dates, vader_norm, label='VADER (normalized)', linewidth=2, color='blue', alpha=0.7)
    ax6.plot(dates, wordbank_norm, label='Word Bank (normalized)', linewidth=2, color='purple', alpha=0.7)
    ax6.set_title('VADER vs Word Bank Sentiment (Normalized)', fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Normalized Score')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'apple_sentiment_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved visualization to: {output_path}")
    
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze sentiment of Apple articles using VADER")
    parser.add_argument(
        "--file",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "apple_articles_compiled2.csv"),
        help="Path to compiled articles CSV"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Chunk size for reading CSV"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "apple_daily_sentiment.csv"),
        help="Output CSV for daily sentiment data"
    )
    args = parser.parse_args()
    
    # Process articles
    daily_df = process_articles_csv(args.file, args.chunksize)
    
    if daily_df is None or len(daily_df) == 0:
        print("[ERROR] No data processed. Exiting.")
        return
    
    # Save daily sentiment data
    daily_df.to_csv(args.output, index=False)
    print(f"\n[OK] Saved daily sentiment data to: {args.output}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SENTIMENT SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Date range: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
    print(f"Total days: {len(daily_df)}")
    print(f"Total articles: {daily_df['Article_Count'].sum():,}")
    print(f"\nVADER Compound Sentiment:")
    print(f"  Mean: {daily_df['VADER_Compound_Mean'].mean():.4f}")
    print(f"  Std: {daily_df['VADER_Compound_Mean'].std():.4f}")
    print(f"  Min: {daily_df['VADER_Compound_Mean'].min():.4f}")
    print(f"  Max: {daily_df['VADER_Compound_Mean'].max():.4f}")
    print(f"\nWord Bank Net Sentiment:")
    print(f"  Mean: {daily_df['WordBank_Net_Sentiment'].mean():.2f}")
    print(f"  Std: {daily_df['WordBank_Net_Sentiment'].std():.2f}")
    print(f"  Min: {daily_df['WordBank_Net_Sentiment'].min():.0f}")
    print(f"  Max: {daily_df['WordBank_Net_Sentiment'].max():.0f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_sentiment_analysis(daily_df)
    
    print("\n[OK] Analysis complete!")

if __name__ == "__main__":
    main()

