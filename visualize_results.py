import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils.metrics import compute_curves
from utils.plotting import plot_roc_pr_cm


def load_simulation_results(results_dir):
    """Load simulation results from CSV."""
    csv_path = os.path.join(results_dir, "simulation_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def prepare_results_df(df):
    """Ensure required columns exist regardless of simulator or realtime outputs."""
    results_df = df.copy()
    
    # Ensure prediction column exists (convert from label names if needed)
    if 'prediction' not in results_df.columns and 'prediction_label' in results_df.columns:
        mapping = {'ATTACK': 1, 'Normal': 0, 'NORMAL': 0}
        results_df['prediction'] = results_df['prediction_label'].map(mapping).fillna(0).astype(int)
    
    # Ensure prediction label column exists
    if 'prediction_label' not in results_df.columns and 'prediction' in results_df.columns:
        results_df['prediction_label'] = np.where(results_df['prediction'] == 1, 'ATTACK', 'NORMAL')
    
    # Normalize true labels (may be stored as strings or missing)
    if 'true_label' in results_df.columns:
        results_df['true_label'] = results_df['true_label'].apply(
            lambda v: np.nan if pd.isna(v) else int(v)
        )
    
    if 'probability' not in results_df.columns:
        raise ValueError("Missing 'probability' column in simulation_results.csv")
    
    return results_df


def create_validation_dashboard(results_df, output_dir):
   
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if validation data is available
    has_labels = has_valid_labels(results_df)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Prediction Distribution
    ax1 = plt.subplot(2, 3, 1)
    results_df['prediction_label'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'red'])
    ax1.set_title('Prediction Distribution')
    ax1.set_xlabel('Prediction')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    # 2. Probability Distribution
    ax2 = plt.subplot(2, 3, 2)
    results_df['probability'].hist(bins=50, ax=ax2, edgecolor='black')
    ax2.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    ax2.set_title('Attack Probability Distribution')
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. Accuracy over time (if labels available)
    if has_labels:
        ax3 = plt.subplot(2, 3, 3)
        # Calculate rolling accuracy
        valid_df = results_df[results_df['true_label'].notna()]
        window_size = min(1000, max(1, len(valid_df) // 10))
        if window_size > 0:
            valid_df = valid_df.copy()
            valid_df['correct'] = valid_df['prediction'] == valid_df['true_label']
            valid_df['rolling_accuracy'] = valid_df['correct'].rolling(window=window_size).mean()
            ax3.plot(valid_df['rolling_accuracy'])
            ax3.set_title(f'Rolling Accuracy (window={window_size})')
            ax3.set_xlabel('Packet Number')
            ax3.set_ylabel('Accuracy')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data\nfor rolling accuracy', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Rolling Accuracy')
    
    # 4. Error Analysis (if labels available)
    if has_labels:
        ax4 = plt.subplot(2, 3, 4)
        if 'error_type' in results_df.columns:
            error_counts = results_df['error_type'].value_counts()
        else:
            label_mask = results_df['true_label'].notna()
            subset = results_df[label_mask]
            error_series = np.where(
                (subset['prediction'] == 1) & (subset['true_label'] == 0),
                'FP',
                np.where(
                    (subset['prediction'] == 0) & (subset['true_label'] == 1),
                    'FN',
                    'Correct'
                )
            )
            error_counts = pd.Series(error_series).value_counts()
        error_counts.plot(kind='bar', ax=ax4, color=['green', 'orange', 'red'])
        ax4.set_title('Error Type Distribution')
        ax4.set_xlabel('Error Type')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. Confusion Matrix (if labels available)
    if has_labels:
        ax5 = plt.subplot(2, 3, 5)
        from sklearn.metrics import confusion_matrix
        label_mask = results_df['true_label'].notna()
        subset = results_df[label_mask]
        if not subset.empty:
            cm = confusion_matrix(subset['true_label'], subset['prediction'], labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                       xticklabels=['Normal', 'Attack'],
                       yticklabels=['Normal', 'Attack'])
            ax5.set_title('Confusion Matrix')
            ax5.set_ylabel('True Label')
            ax5.set_xlabel('Predicted Label')
        else:
            ax5.text(0.5, 0.5, 'No labeled samples', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Confusion Matrix')
    
    # 6. Probability by True Label (if labels available)
    if has_labels:
        ax6 = plt.subplot(2, 3, 6)
        label_mask = results_df['true_label'].notna()
        normal_probs = results_df[(results_df['true_label'] == 0) & label_mask]['probability']
        attack_probs = results_df[(results_df['true_label'] == 1) & label_mask]['probability']
        if len(normal_probs) > 0 or len(attack_probs) > 0:
            ax6.hist([normal_probs, attack_probs], bins=30, label=['Normal', 'Attack'], 
                    alpha=0.7, edgecolor='black')
            ax6.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
            ax6.set_title('Probability Distribution by True Label')
            ax6.set_xlabel('Attack Probability')
            ax6.set_ylabel('Frequency')
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'No labeled samples', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Probability Distribution by True Label')
    
    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, "validation_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Validation dashboard saved to: {dashboard_path}")


def has_valid_labels(results_df):
    return 'true_label' in results_df.columns and results_df['true_label'].notna().any()


def create_performance_report(results_df, output_dir):
    """Create detailed performance report."""
    report_path = os.path.join(output_dir, "performance_report.txt")
    
    with open(report_path, "w") as f:
        f.write("NIDS Performance Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Packets Analyzed: {len(results_df)}\n")
        f.write(f"Predicted Normal: {len(results_df[results_df['prediction'] == 0])} "
                f"({len(results_df[results_df['prediction'] == 0])/len(results_df)*100:.2f}%)\n")
        f.write(f"Predicted Attack: {len(results_df[results_df['prediction'] == 1])} "
                f"({len(results_df[results_df['prediction'] == 1])/len(results_df)*100:.2f}%)\n\n")
        
        f.write(f"Average Attack Probability: {results_df['probability'].mean():.4f}\n")
        f.write(f"Median Attack Probability: {results_df['probability'].median():.4f}\n")
        f.write(f"Min Probability: {results_df['probability'].min():.4f}\n")
        f.write(f"Max Probability: {results_df['probability'].max():.4f}\n\n")
        
        if has_valid_labels(results_df):
            from sklearn.metrics import confusion_matrix
            from utils.metrics import evaluate_binary
            
            label_mask = results_df['true_label'].notna()
            y_true = results_df.loc[label_mask, 'true_label'].astype(int).values
            y_pred = results_df.loc[label_mask, 'prediction'].values
            y_probs = results_df.loc[label_mask, 'probability'].values
            
            metrics = evaluate_binary(y_true, y_probs)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            f.write("VALIDATION METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.6f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {metrics['precision']:.6f} ({metrics['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {metrics['recall']:.6f} ({metrics['recall']*100:.2f}%)\n")
            f.write(f"F1 Score:  {metrics['f1']:.6f} ({metrics['f1']*100:.2f}%)\n\n")
            
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            f.write("CONFUSION MATRIX\n")
            f.write("-"*70 + "\n")
            f.write(f"True Negatives (TN):  {tn:6d}\n")
            f.write(f"False Positives (FP): {fp:6d}\n")
            f.write(f"False Negatives (FN): {fn:6d}\n")
            f.write(f"True Positives (TP):  {tp:6d}\n\n")
            
            if tn + fp > 0:
                f.write(f"False Positive Rate: {fp/(tn+fp)*100:.2f}%\n")
            if fn + tp > 0:
                f.write(f"False Negative Rate: {fn/(fn+tp)*100:.2f}%\n")
    
    print(f"Performance report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize NIDS simulation results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing simulation_results.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for visualizations (default: same as results_dir)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    # Load results
    print(f"Loading simulation results from: {args.results_dir}")
    results_df = load_simulation_results(args.results_dir)
    results_df = prepare_results_df(results_df)
    print(f"Loaded {len(results_df)} packet predictions")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_validation_dashboard(results_df, args.output_dir)
    create_performance_report(results_df, args.output_dir)
    
    # Generate ROC/PR curves if labels available
    if has_valid_labels(results_df):
        print("\nGenerating ROC and PR curves...")
        label_mask = results_df['true_label'].notna()
        y_true = results_df.loc[label_mask, 'true_label'].astype(int).values
        y_probs = results_df.loc[label_mask, 'probability'].values
        curves = compute_curves(y_true, y_probs)
        plot_roc_pr_cm(curves, args.output_dir, title_prefix="NIDS-Simulation")
    
    


if __name__ == "__main__":
    main()

