import pandas as pd

def evaluate_predictions():
    """Evaluate model predictions against ground truth labels"""
    
    # Load actual and predicted values
    labels = pd.read_csv('../data/labels.csv')
    preds = pd.read_csv('predictions.csv')
    
    # Merge on image_id
    merged = labels.merge(preds, on='image_id', suffixes=('_true', '_pred'))
    merged['error'] = abs(merged['hgb_true'] - merged['hgb_pred'])
    
    # Display results
    print("="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    print("\nSample Predictions:")
    print(merged[['image_id', 'hgb_true', 'hgb_pred', 'error']].head(10).to_string(index=False))
    
    print(f"\n{'='*70}")
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"Mean Absolute Error (MAE):  {merged['error'].mean():.4f} g/dL")
    print(f"Median Absolute Error:      {merged['error'].median():.4f} g/dL")
    print(f"Standard Deviation:         {merged['error'].std():.4f} g/dL")
    print(f"Min Error:                  {merged['error'].min():.4f} g/dL")
    print(f"Max Error:                  {merged['error'].max():.4f} g/dL")
    print(f"\nTarget MAE (Competition):   0.8000 g/dL")
    print(f"Current MAE vs Target:      {merged['error'].mean() - 0.8:+.4f} g/dL")
    print("="*70)
    
    # Save detailed results
    merged.to_csv('evaluation_results.csv', index=False)
    print("\nDetailed results saved to: evaluation_results.csv")

if __name__ == '__main__':
    evaluate_predictions()