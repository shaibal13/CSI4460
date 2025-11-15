#!/usr/bin/env python3
"""
Test script to load trained models and make predictions on new data.
Supports all 4 models: Logistic Regression, XGBoost, MLP, and FT-Transformer.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from data_loader import load_unsw_kaggle_dataset
from models.logistic_regression import LogisticRegressionModel
from models.xgboost import XGBoostModel
from models.mlp import MLPTrainer
from models.ft_transformer import FTTransformerTrainer
from utils.metrics import evaluate_binary, compute_curves
from utils.plotting import plot_roc_pr_cm




def load_model(model_name, model_path, device, pos_weight_ratio=1.0, categories=None, num_cont=None):
    """Load a trained model based on model type."""
    print(f"[Test] Loading {model_name} model from: {model_path}")
    
    if model_name == "logreg":
        model = LogisticRegressionModel()
        model.load(model_path)
        return model, None
    
    elif model_name == "xgb":
        model = XGBoostModel(pos_weight_ratio)
        model.load(model_path)
        return model, None
    
    elif model_name == "mlp":
        # For MLP, we need input_dim to create the model
        # Try to load checkpoint to get info, or use a default
        # Actually, we'll need input_dim passed as parameter
        # For now, create with a placeholder - will be set properly in main()
        trainer = MLPTrainer(input_dim=1, pos_weight_ratio=pos_weight_ratio, lr=1e-3)
        # Note: input_dim should match the data shape
        return trainer, None
    
    elif model_name == "ftt":
        if categories is None or num_cont is None:
            raise ValueError("For FTT, categories and num_cont must be provided")
        trainer = FTTransformerTrainer(
            categories=categories,
            num_cont=num_cont,
            pos_weight_ratio=pos_weight_ratio,
            dim=128,
            depth=4,
            heads=8,
            dropout=0.1,
            lr=1e-4
        )
        trainer.load_best(model_path, device)
        return trainer, None
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict(model, model_name, X_full=None, X_num=None, X_cat=None, device=None, batch_size=512):
    """Make predictions using the loaded model."""
    print(f"[Test] Making predictions with {model_name}...")
    
    if model_name in ["logreg", "xgb"]:
        probs = model.predict_proba(X_full)
    
    elif model_name == "mlp":
        probs = model.predict_proba(X_full, device=device, batch_size=batch_size)
    
    elif model_name == "ftt":
        probs = model.predict_proba(X_num, X_cat, device=device, batch_size=batch_size)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return probs


def main():
    parser = argparse.ArgumentParser(description="Load trained models and make predictions")
    
    parser.add_argument("--model", type=str, required=True,
                        choices=["logreg", "xgb", "mlp", "ftt"],
                        help="Model type to load")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory containing UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv")
    parser.add_argument("--output_dir", type=str, default="test_results",
                        help="Directory to save predictions and results")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for MLP/FTT inference")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save prediction probabilities to file")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data using the existing data_loader
    print(f"[Test] Loading data from directory: {args.data_dir}")
    (X_train_full, X_test_full,
     X_train_num, X_test_num,
     X_train_cat, X_test_cat,
     y_train, y_test,
     num_cols, cat_cols,
     scaler) = load_unsw_kaggle_dataset(args.data_dir)
    
    # Use test data for prediction
    X_full = X_test_full
    X_num = X_test_num
    X_cat = X_test_cat
    y_true = y_test
    
    # For FTT, calculate categories from train+test
    if args.model == "ftt":
        if X_train_cat.shape[1] > 0:
            all_cat = np.vstack([X_train_cat, X_test_cat]) if X_test_cat.shape[0] > 0 else X_train_cat
            categories = tuple(int(np.max(all_cat[:, i]) + 1) for i in range(X_train_cat.shape[1]))
            num_cont = X_train_num.shape[1]
            print(f"[Test] FTT categories: {categories}, num_cont: {num_cont}")
        else:
            categories = (2,)
            num_cont = X_train_num.shape[1]
    else:
        categories = None
        num_cont = None
    
    # Calculate pos_weight_ratio from training data (if available)
    pos_weight_ratio = 1.0
    if y_train is not None and len(y_train) > 0:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        pos_weight_ratio = max(1.0, n_neg / max(1, n_pos))
    
    # Load model (MLP needs input_dim, so handle separately)
    if args.model == "mlp":
        input_dim = X_full.shape[1]
        trainer = MLPTrainer(input_dim=input_dim, pos_weight_ratio=pos_weight_ratio, lr=1e-3)
        trainer.load_best(args.model_path, device)
        model = trainer
    else:
        model, _ = load_model(args.model, args.model_path, device, 
                             pos_weight_ratio=pos_weight_ratio,
                             categories=categories, num_cont=num_cont)
    
    # Make predictions
    if args.model in ["logreg", "xgb"]:
        probs = predict(model, args.model, X_full=X_full)
    elif args.model == "mlp":
        probs = predict(model, args.model, X_full=X_full, device=device, batch_size=args.batch_size)
    else:  # ftt
        probs = predict(model, args.model, X_num=X_num, X_cat=X_cat, device=device, batch_size=args.batch_size)
    
    print(f"[Test] Predictions shape: {probs.shape}")
    print(f"[Test] Prediction range: [{np.min(probs):.4f}, {np.max(probs):.4f}]")
    print(f"[Test] Mean prediction: {np.mean(probs):.4f}")
    
    # Evaluate if labels are available
    if y_true is not None:
        print("\n[Test] Evaluating predictions...")
        metrics = evaluate_binary(y_true, probs)
        print(f"[Test] Metrics: {metrics}")
        
        curves = compute_curves(y_true, probs)
        print(f"[Test] ROC-AUC: {curves['roc_auc']:.6f}, PR-AUC: {curves['pr_auc']:.6f}")
        
        # Save plots
        plot_roc_pr_cm(curves, args.output_dir, title_prefix=f"{args.model.upper()}-Test")
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"{args.model.upper()} Test Results\n")
            f.write(f"Metrics: {metrics}\n")
            f.write(f"ROC-AUC: {curves['roc_auc']:.6f}, PR-AUC: {curves['pr_auc']:.6f}\n")
        print(f"[Test] Metrics saved to {metrics_path}")
    else:
        print("[Test] No labels available - skipping evaluation")
    
    # Save predictions if requested
    if args.save_predictions:
        preds_path = os.path.join(args.output_dir, "predictions.npy")
        np.save(preds_path, probs)
        print(f"[Test] Predictions saved to {preds_path}")
        
        # Also save as CSV with predictions
        preds_csv_path = os.path.join(args.output_dir, "predictions.csv")
        pred_df = pd.DataFrame({
            "prediction_probability": probs,
            "prediction": (probs >= 0.5).astype(int)
        })
        pred_df.to_csv(preds_csv_path, index=False)
        print(f"[Test] Predictions CSV saved to {preds_csv_path}")
    
    print("\n[Test] Done!")


if __name__ == "__main__":
    main()

