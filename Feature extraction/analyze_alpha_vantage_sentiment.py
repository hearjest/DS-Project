import pandas as pd
import numpy as np
import os
from datetime import datetime

# VADER Sentiment Analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("[ERROR] vaderSentiment not installed. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

# Import word banks from the main sentiment analysis script
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

def parse_alpha_vantage_time(time_str):
    """Parse Alpha Vantage time format: YYYYMMDDTHHMMSS"""
    try:
        if pd.isna(time_str):
            return None
        time_str = str(time_str).strip()
        # Format: 20251205T130824
        if 'T' in time_str:
            date_part, time_part = time_str.split('T')
            year = int(date_part[:4])
            month = int(date_part[4:6])
            day = int(date_part[6:8])
            hour = int(time_part[:2]) if len(time_part) >= 2 else 0
            minute = int(time_part[2:4]) if len(time_part) >= 4 else 0
            second = int(time_part[4:6]) if len(time_part) >= 6 else 0
            return pd.Timestamp(year, month, day, hour, minute, second)
        else:
            # Try parsing as regular date
            return pd.to_datetime(time_str, errors='coerce')
    except Exception as e:
        return pd.to_datetime(time_str, errors='coerce')

def analyze_alpha_vantage_sentiment(input_file, output_file):
    """Analyze sentiment for Alpha Vantage news sentiment CSV"""
    if not VADER_AVAILABLE:
        print("[ERROR] VADER not available. Cannot proceed.")
        return
    
    print("=" * 80)
    print("ALPHA VANTAGE SENTIMENT ANALYSIS")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Initialize VADER analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Read the CSV
    print("\nReading CSV file...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Loaded {len(df):,} rows")
    
    # Parse time column to datetime
    print("Parsing time column...")
    df['parsed_date'] = df['time'].apply(parse_alpha_vantage_time)
    df['date'] = df['parsed_date'].dt.date if df['parsed_date'].notna().any() else None
    
    # Analyze sentiment for title column
    print("Analyzing sentiment (VADER)...")
    vader_results = df['title'].apply(lambda x: analyze_sentiment_vader(x, analyzer))
    
    df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
    df['vader_pos'] = vader_results.apply(lambda x: x['pos'])
    df['vader_neu'] = vader_results.apply(lambda x: x['neu'])
    df['vader_neg'] = vader_results.apply(lambda x: x['neg'])
    
    print("Analyzing sentiment (Word Bank)...")
    wordbank_results = df['title'].apply(
        lambda x: get_word_bank_sentiment(x, POSITIVE_STOCK_WORDS, NEGATIVE_STOCK_WORDS)
    )
    
    df['wordbank_pos'] = wordbank_results.apply(lambda x: x['positive_words'])
    df['wordbank_neg'] = wordbank_results.apply(lambda x: x['negative_words'])
    df['wordbank_net'] = wordbank_results.apply(lambda x: x['net_sentiment'])
    
    # Reorder columns: keep original columns, add sentiment columns at the end
    original_cols = [col for col in df.columns if not col.startswith('vader_') and 
                     not col.startswith('wordbank_') and col != 'parsed_date']
    sentiment_cols = ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg',
                      'wordbank_pos', 'wordbank_neg', 'wordbank_net']
    
    # If date column was created, add it
    if 'date' in df.columns:
        final_cols = original_cols + ['date'] + sentiment_cols
    else:
        final_cols = original_cols + sentiment_cols
    
    df_output = df[final_cols].copy()
    
    # Save to CSV
    print(f"\nSaving results to: {output_file}")
    df_output.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total articles analyzed: {len(df_output):,}")
    
    if 'date' in df_output.columns:
        valid_dates = df_output['date'].dropna()
        if len(valid_dates) > 0:
            print(f"Date range: {valid_dates.min()} to {valid_dates.max()}")
    
    print(f"\nVADER Sentiment Statistics:")
    print(f"  Compound - Mean: {df_output['vader_compound'].mean():.4f}, Std: {df_output['vader_compound'].std():.4f}")
    print(f"  Positive - Mean: {df_output['vader_pos'].mean():.4f}, Std: {df_output['vader_pos'].std():.4f}")
    print(f"  Neutral  - Mean: {df_output['vader_neu'].mean():.4f}, Std: {df_output['vader_neu'].std():.4f}")
    print(f"  Negative - Mean: {df_output['vader_neg'].mean():.4f}, Std: {df_output['vader_neg'].std():.4f}")
    
    print(f"\nWord Bank Statistics:")
    print(f"  Positive words - Mean: {df_output['wordbank_pos'].mean():.2f}, Max: {df_output['wordbank_pos'].max():.0f}")
    print(f"  Negative words - Mean: {df_output['wordbank_neg'].mean():.2f}, Max: {df_output['wordbank_neg'].max():.0f}")
    print(f"  Net sentiment  - Mean: {df_output['wordbank_net'].mean():.2f}, Range: [{df_output['wordbank_net'].min():.0f}, {df_output['wordbank_net'].max():.0f}]")
    
    print(f"\n[OK] Analysis complete! Results saved to: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze sentiment for Alpha Vantage news sentiment CSV")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpha_vantage_news_sentiment.csv"),
        help="Path to Alpha Vantage news sentiment CSV"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpha_vantage_news_sentiment_with_vader.csv"),
        help="Output CSV path"
    )
    args = parser.parse_args()
    
    analyze_alpha_vantage_sentiment(args.input, args.output)

if __name__ == "__main__":
    main()

