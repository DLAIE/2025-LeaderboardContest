#!/usr/bin/env python3
"""
generate_leaderboard.py
Reads submissions_full.csv, ranks teams, generates leaderboard.csv
"""
import pandas as pd
import argparse

def generate_leaderboard(submissions_file='submissions_full.csv', output_file='leaderboard.csv'):
    """Generate leaderboard from submissions file."""
    
    # Read submissions
    df = pd.read_csv(submissions_file)
    
    # Keep only most recent submission per team
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.sort_values('time_stamp').groupby('team').tail(1)
    
    # Select columns for leaderboard
    metrics_cols = ['total_params', 'time_per_sample', 'mse', 'ssim', 
                    'entropy', 'kl_div_classes', 'gen_confidence']
    keep_cols = ['team', 'names', 'time_stamp'] + metrics_cols
    df = df[keep_cols].copy()
    
    # Rank each metric (lower rank = better)
    # For metrics where lower is better: ascending=True
    # For metrics where higher is better: ascending=False
    rank_ascending = {
        'total_params': True,      # ↓
        'time_per_sample': True,   # ↓
        'mse': True,               # ↓
        'ssim': False,             # ↑
        'entropy': True,           # ↓
        'kl_div_classes': True,    # ↓
        'gen_confidence': False    # ↑
    }
    
    for col in metrics_cols:
        rank_col = f'{col}_rank'
        df[rank_col] = df[col].rank(ascending=rank_ascending[col], method='min')
    
    # Calculate overall rank (average of all ranks) - add as rightmost column
    rank_cols = [f'{col}_rank' for col in metrics_cols]
    df['overall_rank'] = df[rank_cols].mean(axis=1)
    
    # Sort by overall rank
    df = df.sort_values('overall_rank')
    
    # Reorder columns to put overall_rank at the end
    other_cols = [col for col in df.columns if col != 'overall_rank']
    df = df[other_cols + ['overall_rank']]
    
    # Save leaderboard
    df.to_csv(output_file, index=False)
    print(f"Leaderboard saved to {output_file}")
    
    # Display summary
    print("\n" + "="*100)
    print("LEADERBOARD")
    print("="*100)
    display_cols = ['team'] + ['time_stamp']+ metrics_cols + ['overall_rank']
    print(df[display_cols].to_string(index=False))
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='submissions_full.csv', help='Input submissions file')
    parser.add_argument('--output', default='leaderboard.csv', help='Output leaderboard file')
    args = parser.parse_args()
    
    generate_leaderboard(args.input, args.output)