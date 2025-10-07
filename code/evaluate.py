import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(errors):
    """Calculate comprehensive error metrics"""
    return {
        'mae': errors.mean(),
        'median_ae': errors.median(),
        'std': errors.std(),
        'min': errors.min(),
        'max': errors.max(),
        'q25': errors.quantile(0.25),
        'q75': errors.quantile(0.75),
        'rmse': np.sqrt((errors ** 2).mean())
    }

def evaluate_predictions(labels_path, preds_path, output_dir='./'):
    """Evaluate model predictions against ground truth labels"""
    
    # Load actual and predicted values
    labels = pd.read_csv(labels_path)
    preds = pd.read_csv(preds_path)
    
    # Merge on image_id
    merged = labels.merge(preds, on='image_id', suffixes=('_true', '_pred'))
    
    # Handle column naming (support both 'hgb' and 'hgb_pred' formats)
    if 'hgb_true' in merged.columns and 'hgb_pred' in merged.columns:
        true_col = 'hgb_true'
        pred_col = 'hgb_pred'
    elif 'hgb' in merged.columns and 'hgb_pred' in merged.columns:
        merged.rename(columns={'hgb': 'hgb_true'}, inplace=True)
        true_col = 'hgb_true'
        pred_col = 'hgb_pred'
    else:
        print("ERROR: Could not find correct columns in merged data")
        print(f"Available columns: {merged.columns.tolist()}")
        return
    
    merged['error'] = abs(merged[true_col] - merged[pred_col])
    merged['squared_error'] = (merged[true_col] - merged[pred_col]) ** 2
    merged['relative_error'] = merged['error'] / merged[true_col] * 100
    
    # Display results
    print("="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    print("\nSample Predictions:")
    display_cols = ['image_id', true_col, pred_col, 'error']
    print(merged[display_cols].head(10).to_string(index=False))
    
    # Calculate metrics
    metrics = calculate_metrics(merged['error'])
    
    print(f"\n{'='*70}")
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"Mean Absolute Error (MAE):     {metrics['mae']:.4f} g/dL")
    print(f"Median Absolute Error:         {metrics['median_ae']:.4f} g/dL")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f} g/dL")
    print(f"Standard Deviation:            {metrics['std']:.4f} g/dL")
    print(f"Min Error:                     {metrics['min']:.4f} g/dL")
    print(f"Max Error:                     {metrics['max']:.4f} g/dL")
    print(f"25th Percentile:               {metrics['q25']:.4f} g/dL")
    print(f"75th Percentile:               {metrics['q75']:.4f} g/dL")
    print(f"\nMean Relative Error:           {merged['relative_error'].mean():.2f}%")
    
    print(f"\n{'='*70}")
    print("TARGET COMPARISON")
    print("="*70)
    target_mae = 0.8
    print(f"Target MAE (Competition):      {target_mae:.4f} g/dL")
    print(f"Current MAE:                   {metrics['mae']:.4f} g/dL")
    print(f"Difference from Target:        {metrics['mae'] - target_mae:+.4f} g/dL")
    
    if metrics['mae'] <= target_mae:
        print(f"✓ TARGET ACHIEVED! ({(target_mae - metrics['mae']):.4f} g/dL below target)")
    else:
        print(f"✗ Need to improve by:          {metrics['mae'] - target_mae:.4f} g/dL")
    
    # Error distribution analysis
    print(f"\n{'='*70}")
    print("ERROR DISTRIBUTION")
    print("="*70)
    bins = [0, 0.5, 1.0, 1.5, 2.0, float('inf')]
    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '>2.0']
    merged['error_bin'] = pd.cut(merged['error'], bins=bins, labels=labels)
    error_dist = merged['error_bin'].value_counts().sort_index()
    
    for bin_label, count in error_dist.items():
        pct = count / len(merged) * 100
        print(f"  {bin_label} g/dL: {count:3d} samples ({pct:5.1f}%)")
    
    # Worst predictions
    print(f"\n{'='*70}")
    print("WORST PREDICTIONS (Top 5)")
    print("="*70)
    worst = merged.nlargest(5, 'error')[['image_id', true_col, pred_col, 'error']]
    print(worst.to_string(index=False))
    
    # Best predictions
    print(f"\n{'='*70}")
    print("BEST PREDICTIONS (Top 5)")
    print("="*70)
    best = merged.nsmallest(5, 'error')[['image_id', true_col, pred_col, 'error']]
    print(best.to_string(index=False))
    
    # Predictions by HgB range
    print(f"\n{'='*70}")
    print("PERFORMANCE BY HGB RANGE")
    print("="*70)
    hgb_bins = [0, 7, 10, 13, float('inf')]
    hgb_labels = ['Severe (<7)', 'Moderate (7-10)', 'Mild (10-13)', 'Normal (>13)']
    merged['hgb_range'] = pd.cut(merged[true_col], bins=hgb_bins, labels=hgb_labels)
    
    for range_label in hgb_labels:
        range_data = merged[merged['hgb_range'] == range_label]
        if len(range_data) > 0:
            range_mae = range_data['error'].mean()
            print(f"  {range_label:20s}: MAE = {range_mae:.4f} g/dL (n={len(range_data):2d})")
    
    print("="*70)
    
    # Save detailed results
    output_csv = f"{output_dir}/evaluation_results.csv"
    merged.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")
    
    # Create visualization if matplotlib available
    try:
        create_visualizations(merged, true_col, pred_col, output_dir)
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
    
    return metrics

def create_visualizations(merged, true_col, pred_col, output_dir):
    """Create evaluation visualizations"""
    print(f"\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot: Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(merged[true_col], merged[pred_col], alpha=0.6, s=100)
    ax1.plot([merged[true_col].min(), merged[true_col].max()], 
             [merged[true_col].min(), merged[true_col].max()], 
             'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True HgB (g/dL)', fontsize=12)
    ax1.set_ylabel('Predicted HgB (g/dL)', fontsize=12)
    ax1.set_title('Predicted vs Actual HgB', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(merged['error'], bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(merged['error'].mean(), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {merged["error"].mean():.2f}')
    ax2.axvline(merged['error'].median(), color='g', linestyle='--', 
                linewidth=2, label=f'Median: {merged["error"].median():.2f}')
    ax2.set_xlabel('Absolute Error (g/dL)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual plot
    ax3 = axes[1, 0]
    residuals = merged[pred_col] - merged[true_col]
    ax3.scatter(merged[true_col], residuals, alpha=0.6, s=100)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('True HgB (g/dL)', fontsize=12)
    ax3.set_ylabel('Residuals (g/dL)', fontsize=12)
    ax3.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error by true HgB value
    ax4 = axes[1, 1]
    ax4.scatter(merged[true_col], merged['error'], alpha=0.6, s=100)
    ax4.set_xlabel('True HgB (g/dL)', fontsize=12)
    ax4.set_ylabel('Absolute Error (g/dL)', fontsize=12)
    ax4.set_title('Error by True HgB Value', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate hemoglobin predictions')
    parser.add_argument('--labels', type=str, default='../data/labels.csv', 
                        help='Path to ground truth labels CSV')
    parser.add_argument('--preds', type=str, default='predictions.csv',
                        help='Path to predictions CSV')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    evaluate_predictions(args.labels, args.preds, args.output_dir)