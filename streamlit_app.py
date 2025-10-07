#!/usr/bin/env python3



# Usage:
# Local testing:
# $ ./streamlit_app.py --no-streamlit

# Streamlit mode (also works when deployed):
# $ streamlit run streamlit_app.py

# When deployed to Streamlit Cloud, it'll ignore the argparse stuff and just run normally.



import pandas as pd
from datetime import datetime
import argparse

def display_leaderboard(use_streamlit=True):
    """Display leaderboard, optionally using Streamlit."""
    
    df = pd.read_csv('leaderboard.csv')
    df = df.drop(columns=['names'], errors='ignore')  # keep names private
    # Drop individual rank columns - users can sort interactively
    rank_cols = [col for col in df.columns if col.endswith('_rank') and col != 'overall_rank']
    df_display = df.drop(columns=rank_cols)
    
    if not use_streamlit:
        print("\n" + "="*100)
        print("LEADERBOARD (Raw CSV columns)")
        print("="*100)
        print(df.to_string(index=False))
    
    # Rename columns for display
    df_display = df_display.rename(columns={
        'team': 'Team Name',
        'overall_rank': 'Overall Rank',
        'total_params': 'Total Params ↓',
        'time_per_sample': 'Time/Sample ↓',
        'mse': 'MSE ↓',
        'ssim': 'SSIM ↑',
        'entropy': 'Entropy ↑',
        'kl_div_classes': 'KL Div ↓',
        'gen_confidence': 'Gen Confidence ↑'
    })
    
    if not use_streamlit:
        print("\n" + "="*100)
        print("LEADERBOARD (Display columns)")
        print("="*100)
        print(df_display.to_string(index=False))
        return df_display
    
    # Streamlit mode
    import streamlit as st
    
    st.title("🏆 2025 DLAIE Latent Flow Matching Leaderboard")
    
    # Get latest submission time from the data
    latest_update = pd.to_datetime(df['time_stamp']).max()
    st.caption(f"Last updated: {latest_update.strftime('%Y-%m-%d %H:%M')}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    return df_display

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display leaderboard")
    parser.add_argument('--no-streamlit', action='store_true', 
                       help='Run in CLI mode for local testing')
    args = parser.parse_args()
    
    display_leaderboard(use_streamlit=not args.no_streamlit)