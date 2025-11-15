# #!/usr/bin/env python3
# import os
# import argparse
# from typing import List, Dict, Tuple

# import json
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
#                              precision_recall_curve, auc, f1_score, precision_score, recall_score)
# from sklearn.metrics import roc_curve
# from sklearn.model_selection import train_test_split

# import platform
# import warnings
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# # Suppress sklearn warnings for single-class cases
# warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')

# # =======================
# #  FT-Transformer Implementation from Paper:
# # "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)
# # https://arxiv.org/abs/2106.11959
# #
# # We use Lucidrains' implementation of this model:
# # https://github.com/lucidrains/tab-transformer-pytorch
# # =======================
# from tab_transformer_pytorch import FTTransformer


# # --------------------------
# # Data loading and preprocessing (UPDATED & FIXED FOR BINARY IDS)
# # --------------------------
# def load_unsw_full_dataset(dataset_path: str) -> pd.DataFrame:
#     """
#     Loads UNSW-NB15 from the 4 CSV parts and assigns column names from NUSW-NB15_features.csv
#     Then returns a single combined dataframe (features + label column).
#     """
#     print("\nLoading UNSW-NB15 full dataset (4 parts)...")

#     # Load feature names (features file sometimes has headers; try common patterns)
#     features_candidates = ["NUSW-NB15_features.csv", "UNSW-NB15_features.csv", "UNSW-NB15_LIST_EVENTS.csv"]
#     features_path = None
#     for c in features_candidates:
#         p = os.path.join(dataset_path, c)
#         if os.path.exists(p):
#             features_path = p
#             break
#     if features_path is None:
#         raise FileNotFoundError("Could not find feature-name CSV in dataset directory. Checked: " + str(features_candidates))

#     # Try reading features file flexibly
#     try:
#         feature_df = pd.read_csv(features_path, header=None, encoding="latin-1")
#         # many versions list feature names in second column
#         if feature_df.shape[1] >= 2:
#             feature_names = feature_df.iloc[:, 1].astype(str).tolist()
#         else:
#             feature_names = feature_df.iloc[:, 0].astype(str).tolist()
#     except Exception:
#         # fallback: read with headers
#         feature_df = pd.read_csv(features_path, encoding="latin-1")
#         if "Name" in feature_df.columns:
#             feature_names = feature_df["Name"].astype(str).tolist()
#         else:
#             feature_names = feature_df.iloc[:, 0].astype(str).tolist()

#     print(f"Loaded {len(feature_names)} feature names from: {features_path}")

#     # Load all 4 dataset chunks (expected filenames)
#     part_files = [
#         "UNSW-NB15_1.csv",
#         "UNSW-NB15_2.csv",
#         "UNSW-NB15_3.csv",
#         "UNSW-NB15_4.csv"
#     ]

#     df_list = []
#     for p in part_files:
#         fp = os.path.join(dataset_path, p)
#         if not os.path.exists(fp):
#             print(f"Warning: part file not found: {fp} (skipping)")
#             continue
#         print(f"Reading: {fp}")
#         df_part = pd.read_csv(fp, header=None, names=feature_names, low_memory=False)
#         df_list.append(df_part)

#     if len(df_list) == 0:
#         raise FileNotFoundError("No UNSW part CSVs found in dataset directory.")

#     full_df = pd.concat(df_list, ignore_index=True)
#     print(" Full dataset loaded:", full_df.shape)

#     # Attempt to locate label column (common names)
#     label_col = None
#     for col in ["Label", "label", "Label "]:
#         if col in full_df.columns:
#             label_col = col
#             break

#     if label_col is None:
#         # If feature file did not include label, maybe last column is label
#         # try heuristics: if last column contains only 0/1 or small ints, treat as label
#         last_col = full_df.columns[-1]
#         if set(np.unique(full_df[last_col].dropna())) <= {0, 1}:
#             label_col = last_col
#             print(f"Heuristic: using last column '{last_col}' as label.")
#         else:
#             raise KeyError("Could not find label column in dataset. Columns: " + str(full_df.columns.tolist()))

#     # Convert label to integer (0 normal, 1 attack)
#     # full_df[label_col] = full_df[label_col].astype(int)
#     full_df[label_col] = (
#     pd.to_numeric(full_df[label_col], errors="coerce")   # force numeric (invalid → NaN)
#         .fillna(0)                                        # treat missing labels as normal
#         .replace([np.inf, -np.inf], 0)                    # replace infinities
#         .astype(int)
# )
#     # Print distribution
#     label_counts = full_df[label_col].value_counts()
#     print(f"Raw label distribution (from '{label_col}'): {label_counts.to_dict()}")

#     # Extract labels and remove label & attack_cat from features to avoid leakage
#     if "attack_cat" in full_df.columns:
#         print("Dropping 'attack_cat' (attack category) to prevent leakage.")
#     if label_col != "label":
#         # If label column named differently, standardize
#         full_df = full_df.rename(columns={label_col: "label"})

#     # ALWAYS drop attack_cat and label from input features (we will keep label separately)
#     drop_cols = []
#     if "attack_cat" in full_df.columns:
#         drop_cols.append("attack_cat")
#     if "label" in full_df.columns:
#         # keep a copy of labels as column 'label' in the same dataframe for splitting convenience,
#         # but we will extract target BEFORE building feature columns later
#         # here we keep it, but will remove from features before model input
#         pass

#     return full_df


# def identify_columns(df: pd.DataFrame, label_col="label") -> Tuple[List[str], List[str]]:
#     """
#     Detect categorical and numeric columns automatically.
#     Categorical columns are kept for embeddings (no one-hot).
#     Numeric columns will be scaled (RobustScaler).
#     """
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     # Remove label from numeric features if present (we will separate target)
#     if label_col in num_cols:
#         num_cols.remove(label_col)
#     # Also ensure if label is in cat_cols, remove it
#     if label_col in cat_cols:
#         cat_cols.remove(label_col)
#     return cat_cols, num_cols


# def build_category_maps(train_df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict]:
#     """
#     Build mapping for categorical columns from training data only.
#     We map categories -> 1..N and reserve 0 for unknown/unseen.
#     Return a dict: {col: {"mapping": {...}, "size": N+1}}
#     """
#     cat_maps = {}
#     for c in cat_cols:
#         uniques = train_df[c].fillna("__nan__").astype(str).unique().tolist()
#         mapping = {v: i + 1 for i, v in enumerate(sorted(uniques))}
#         cat_maps[c] = {"mapping": mapping, "size": len(mapping) + 1}
#     return cat_maps


# def map_categorical(df: pd.DataFrame, cat_cols: List[str], cat_maps: Dict[str, Dict]) -> pd.DataFrame:
#     """
#     Map categorical values using the train-built maps; unseen -> 0.
#     """
#     out = df.copy()
#     for c in cat_cols:
#         m = cat_maps[c]["mapping"]
#         out[c] = out[c].fillna("__nan__").astype(str).map(lambda v: m.get(v, 0)).astype(int)
#     return out


# # --------------------------
# # Dataset
# # --------------------------
# class IDS_Dataset(Dataset):
#     def __init__(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], label_col="label"):
#         # Note: df here should NOT contain 'label' column as feature
#         self.X_cat = df[cat_cols].astype(np.int64).values if cat_cols else np.zeros((len(df), 0))
#         self.X_num = df[num_cols].astype(np.float32).values if num_cols else np.zeros((len(df), 0))
#         self.y = df[label_col].astype(np.float32).values

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return {
#             "cat": torch.tensor(self.X_cat[idx]) if self.X_cat.size else torch.zeros(0, dtype=torch.long),
#             "num": torch.tensor(self.X_num[idx]) if self.X_num.size else torch.zeros(0, dtype=torch.float32),
#             "y": torch.tensor(self.y[idx], dtype=torch.float32)
#         }


# # --------------------------
# # Combo loss: weighted BCE + focal modulation
# # --------------------------
# def combo_focal_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float = 1.0, gamma=2.0, alpha=0.25):
#     if logits.dim() > 1 and logits.size(1) == 1:
#         logits = logits.squeeze(1)

#     bce = nn.functional.binary_cross_entropy_with_logits(
#         logits, targets, reduction='none', pos_weight=torch.tensor(pos_weight, device=logits.device)
#     )

#     probs = torch.sigmoid(logits)
#     p_t = probs * targets + (1 - probs) * (1 - targets)
#     mod = (1.0 - p_t) ** gamma
#     alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
#     loss = alpha_factor * mod * bce

#     return loss.mean()


# # --------------------------
# # Metric helpers
# # --------------------------
# def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold=0.5) -> Dict:
#     # Check for NaN values and handle them
#     if y_probs.size == 0:
#         return {"roc_auc": float("nan"), "pr_auc": float("nan"),
#                 "f1": 0.0, "precision": 0.0, "recall": 0.0,
#                 "confusion_matrix": np.array([[0, 0], [0, 0]])}

#     nan_mask = np.isnan(y_probs) | np.isinf(y_probs)
#     if np.any(nan_mask):
#         y_probs = np.where(nan_mask, 0.5, y_probs)

#     y_probs = np.clip(y_probs, 0.0, 1.0)
#     y_pred = (y_probs >= threshold).astype(int)

#     unique_labels = np.unique(y_true)
#     has_both_classes = len(unique_labels) > 1
    
#     # Warn if only one class is present
#     if not has_both_classes:
#         print(f"Warning: Only one class ({unique_labels[0]}) found in y_true. Metrics may be unreliable.")
    
#     # ROC-AUC requires both classes
#     if has_both_classes:
#         try:
#             roc = roc_auc_score(y_true, y_probs)
#         except Exception:
#             roc = float("nan")
#     else:
#         roc = float("nan")

#     # PR-AUC: check if both classes exist
#     if has_both_classes:
#         try:
#             p, r, _ = precision_recall_curve(y_true, y_probs)
#             pr_auc = auc(r, p)
#         except Exception:
#             pr_auc = float("nan")
#     else:
#         pr_auc = float("nan")

#     # Always specify labels for confusion matrix to avoid warnings
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
#     # Ensure confusion matrix is 2x2 even if only one class is present
#     if cm.shape == (1, 1):
#         if unique_labels[0] == 0:
#             cm = np.array([[cm[0, 0], 0], [0, 0]])
#         else:
#             cm = np.array([[0, 0], [0, cm[0, 0]]])
#     elif cm.shape == (2, 1) or cm.shape == (1, 2):
#         # Handle edge case where one dimension is missing
#         cm = np.array([[0, 0], [0, 0]])
#         if len(unique_labels) == 1:
#             if unique_labels[0] == 0:
#                 cm[0, 0] = len(y_true[y_true == 0])
#             else:
#                 cm[1, 1] = len(y_true[y_true == 1])

#     return {
#         "roc_auc": roc,
#         "pr_auc": pr_auc,
#         "f1": f1_score(y_true, y_pred, zero_division=0, labels=[0, 1]),
#         "precision": precision_score(y_true, y_pred, zero_division=0, labels=[0, 1]),
#         "recall": recall_score(y_true, y_pred, zero_division=0, labels=[0, 1]),
#         "confusion_matrix": cm
#     }


# # --------------------------
# # Training / evaluation loop
# # --------------------------
# def run_epoch(model, dataloader, optimizer, device, is_train: bool, pos_weight: float, gamma: float, alpha: float):
#     if is_train:
#         model.train()
#     else:
#         model.eval()

#     losses = []
#     all_probs = []
#     all_trues = []

#     for batch in tqdm(dataloader, desc=("Train" if is_train else "Eval"), leave=False):
#         x_cat = batch["cat"].to(device) if batch["cat"].numel() else None
#         x_num = batch["num"].to(device) if batch["num"].numel() else None
#         y = batch["y"].to(device)

#         with torch.set_grad_enabled(is_train):
#             logits = model(x_cat, x_num)
#             if logits.dim() > 1 and logits.size(1) == 1:
#                 logits = logits.squeeze(1)

#             # sanity: check for NaN and clamp extreme logits
#             if torch.isnan(logits).any():
#                 print("Warning: NaN in logits; skipping batch.")
#                 continue
#             logits = torch.clamp(logits, min=-20.0, max=20.0)

#             loss = combo_focal_loss(logits, y, pos_weight=pos_weight, gamma=gamma, alpha=alpha)

#             if torch.isnan(loss) or torch.isinf(loss):
#                 print("Warning: NaN/Inf loss; skipping batch.")
#                 continue

#             if is_train:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()

#         losses.append(loss.item())
#         probs = torch.sigmoid(logits).detach().cpu().numpy()
#         probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
#         all_probs.append(probs)
#         all_trues.append(y.detach().cpu().numpy())

#     if len(losses) == 0:
#         return float("inf"), np.array([]), np.array([])

#     return float(np.mean(losses)), np.concatenate(all_probs), np.concatenate(all_trues)


# # --------------------------
# # Plotting utilities
# # --------------------------
# def plot_loss(train_losses, val_losses, out_dir):
#     plt.figure()
#     plt.plot(train_losses, label="train_loss")
#     plt.plot(val_losses, label="val_loss")
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.legend()
#     plt.title("Training / Validation Loss")
#     p = os.path.join(out_dir, "loss_train_val.png")
#     plt.savefig(p, bbox_inches="tight")
#     plt.close()
#     return p


# def plot_roc(y_true, y_probs, out_dir):
#     if y_probs.size == 0:
#         return None
#     fpr, tpr, _ = roc_curve(y_true, y_probs)
#     auc_score = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else float("nan")
#     plt.figure()
#     plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.4f})")
#     plt.plot([0, 1], [0, 1], "--", color="gray")
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.legend()
#     plt.title("ROC Curve")
#     p = os.path.join(out_dir, "roc_curve.png")
#     plt.savefig(p, bbox_inches="tight")
#     plt.close()
#     return p


# def plot_pr(y_true, y_probs, out_dir):
#     if y_probs.size == 0:
#         return None
#     precision, recall, _ = precision_recall_curve(y_true, y_probs)
#     pr_auc = auc(recall, precision)
#     plt.figure()
#     plt.plot(recall, precision, label=f"PR (AUC={pr_auc:.4f})")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.legend()
#     plt.title("Precision-Recall Curve")
#     p = os.path.join(out_dir, "pr_curve.png")
#     plt.savefig(p, bbox_inches="tight")
#     plt.close()
#     return p


# def plot_confusion(cm, out_dir):
#     plt.figure(figsize=(4, 4))
#     im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     ticks = [0, 1]
#     plt.xticks(ticks, ticks)
#     plt.yticks(ticks, ticks)
#     plt.ylabel("True")
#     plt.xlabel("Predicted")
#     thresh = cm.max() / 2. if cm.max() != 0 else 1.0
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             plt.text(j, i, format(int(cm[i, j]), "d"),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#     p = os.path.join(out_dir, "confusion_matrix.png")
#     plt.savefig(p, bbox_inches="tight")
#     plt.close()
#     return p


# # --------------------------
# # Main
# # --------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Train FT-Transformer (lucidrains) on UNSW-NB15 (binary IDS)")

#     parser.add_argument("--dataset_path", type=str, default="D:/CSI4460/Datasets/UNSW",
#                         help="Path to dataset directory containing UNSW-NB15 CSVs")
#     parser.add_argument("--save_results", type=str, default="results/UNSW",
#                         help="Directory to save metrics, plots, and model checkpoints")

#     parser.add_argument("--epochs", type=int, default=30)
#     parser.add_argument("--batch", type=int, default=256)
#     parser.add_argument("--dim", type=int, default=128)
#     parser.add_argument("--depth", type=int, default=3)
#     parser.add_argument("--heads", type=int, default=8)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--dropout", type=float, default=0.1)

#     parser.add_argument("--gamma", type=float, default=2.0)
#     parser.add_argument("--alpha", type=float, default=0.25)

#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--seed", type=int, default=42)

#     args = parser.parse_args()

#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     os.makedirs(args.save_results, exist_ok=True)

#     # -------------------------
#     # Load FULL UNSW DATASET
#     # -------------------------
#     full_df = load_unsw_full_dataset(args.dataset_path)

#     # Extract label and remove label & attack_cat from features immediately (prevent leakage)
#     if "label" not in full_df.columns:
#         raise KeyError("label column not found after loading dataset.")
#     y_all = full_df["label"].astype(int).values
#     # Make a copy of dataframe for features and drop label/attack_cat
#     features_df = full_df.copy()
#     if "attack_cat" in features_df.columns:
#         features_df = features_df.drop(columns=["attack_cat"], errors="ignore")
#     # drop label column from features (we will keep label separately)
#     features_df = features_df.drop(columns=["label"], errors="ignore")

#     # Debug: show columns used as features (must NOT include label or attack_cat)
#     print("\n[DEBUG] Feature columns used (sample 10):")
#     print(features_df.columns.tolist()[:10], "...", len(features_df.columns), "columns total")

#     # Add back label column to features_df temporarily only for stratified split convenience
#     features_df["label"] = y_all

#     #  Stratified split: train / val / test (using features_df which contains label for splitting)
#     train_df, test_df = train_test_split(features_df, test_size=0.20,
#                                          random_state=args.seed, stratify=features_df["label"])
#     train_df, val_df = train_test_split(train_df, test_size=0.20,
#                                         random_state=args.seed, stratify=train_df["label"])

#     # After split, separate label from feature frames
#     train_labels = train_df["label"].astype(int).values
#     val_labels = val_df["label"].astype(int).values
#     test_labels = test_df["label"].astype(int).values

#     train_df = train_df.drop(columns=["label"])
#     val_df = val_df.drop(columns=["label"])
#     test_df = test_df.drop(columns=["label"])

#     print(f"\nTrain: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
#     # print label distributions
#     print("Train label distribution:", np.bincount(train_labels))
#     print("Val label distribution:", np.bincount(val_labels))
#     print("Test label distribution:", np.bincount(test_labels))

#     # Identify categorical / numeric columns
#     # build on the train_df (which no longer contains label)
#     cat_cols, num_cols = identify_columns(pd.concat([train_df, val_df, test_df], ignore_index=True))
#     print("Categorical columns count:", len(cat_cols))
#     print("Numeric columns count:", len(num_cols))

#     # Build categorical maps using TRAIN ONLY (avoid leakage)
#     # note: train_df still contains raw categorical strings if present
#     train_df_for_maps = train_df.copy()
#     train_df_for_maps[cat_cols] = train_df_for_maps[cat_cols].astype(object).fillna("__nan__").astype(str)

#     cat_maps = build_category_maps(train_df_for_maps, cat_cols)

#     # Map categorical values (train/val/test) using cat_maps; unseen -> 0
#     train_mapped = map_categorical(train_df, cat_cols, cat_maps)
#     val_mapped = map_categorical(val_df, cat_cols, cat_maps)
#     test_mapped = map_categorical(test_df, cat_cols, cat_maps)

#     # Insert label column back for dataset construction (but NOT as feature)
#     train_mapped["label"] = train_labels
#     val_mapped["label"] = val_labels
#     test_mapped["label"] = test_labels

#     # Handle NaN/Inf for numeric columns BEFORE scaling
#     print("\nChecking NaN/Inf in numeric columns...")
#     for col in num_cols:
#         # fill NaN with training median
#         if train_mapped[col].isna().any():
#             fillv = train_mapped[col].median()
#             if pd.isna(fillv):
#                 fillv = 0.0
#             train_mapped[col] = train_mapped[col].fillna(fillv)
#         if val_mapped[col].isna().any():
#             val_mapped[col] = val_mapped[col].fillna(train_mapped[col].median())
#         if test_mapped[col].isna().any():
#             test_mapped[col] = test_mapped[col].fillna(train_mapped[col].median())

#         # replace infinities
#         train_mapped[col] = train_mapped[col].replace([np.inf, -np.inf], [1e6, -1e6])
#         val_mapped[col] = val_mapped[col].replace([np.inf, -np.inf], [1e6, -1e6])
#         test_mapped[col] = test_mapped[col].replace([np.inf, -np.inf], [1e6, -1e6])

#     # Robust scaling numeric columns
#     scaler = RobustScaler()
#     if len(num_cols) > 0:
#         train_mapped[num_cols] = scaler.fit_transform(train_mapped[num_cols])
#         val_mapped[num_cols] = scaler.transform(val_mapped[num_cols])
#         test_mapped[num_cols] = scaler.transform(test_mapped[num_cols])

#     # Final check: no NaN/Inf after preprocessing
#     print("Final NaN counts (train):", train_mapped[num_cols].isna().sum().sum() if num_cols else 0)

#     # -------------------------
#     # DataLoaders
#     # -------------------------
#     # On Windows, multiprocessing with DataLoader can cause issues -> set num_workers=0
#     if platform.system() == 'Windows':
#         num_workers = 0
#         pin_memory = False
#         print("Windows detected: Using num_workers=0 to avoid multiprocessing issues")
#     else:
#         num_workers = args.num_workers
#         pin_memory = True

#     train_ds = IDS_Dataset(train_mapped, cat_cols, num_cols, label_col="label")
#     val_ds = IDS_Dataset(val_mapped, cat_cols, num_cols, label_col="label")
#     test_ds = IDS_Dataset(test_mapped, cat_cols, num_cols, label_col="label")

#     train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
#     val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
#     test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

#     # -------------------------
#     # Model initialization
#     # -------------------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("\nUsing device:", device)

#     # Categories sizes for FTTransformer
#     if len(cat_cols) == 0:
#         print("Warning: No categorical columns found. Using dummy category size.")
#         categories = (2,)
#     else:
#         categories = tuple(cat_maps[c]["size"] for c in cat_cols)

#     print(f"Model config: categories={categories}, num_continuous={len(num_cols)}")

#     model = FTTransformer(
#         categories=categories,
#         num_continuous=len(num_cols),
#         dim=args.dim,
#         depth=args.depth,
#         heads=args.heads,
#         attn_dropout=args.dropout,
#         ff_dropout=args.dropout,
#         dim_out=1
#     ).to(device)

#     # Weight init
#     for param in model.parameters():
#         if param.dim() > 1:
#             torch.nn.init.xavier_uniform_(param)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

#     # compute pos_weight for BCE
#     y_train = train_mapped["label"].values
#     n_pos = int((y_train == 1).sum())
#     n_neg = int((y_train == 0).sum())
#     pos_weight = float(max(1.0, n_neg / max(1, n_pos)))
#     print("pos_weight (neg/pos):", pos_weight)

#     # logging & checkpointing
#     train_losses = []
#     val_losses = []
#     train_log_lines = []
#     best_f1 = -1.0
#     best_ckpt_path = None

#     # -------------------------
#     # Training loop
#     # -------------------------
#     for epoch in range(1, args.epochs + 1):
#         print(f"\n--- Epoch {epoch}/{args.epochs} ---")

#         train_loss, train_probs, train_trues = run_epoch(model, train_loader, optimizer, device, True, pos_weight, args.gamma, args.alpha)
#         val_loss, val_probs, val_trues = run_epoch(model, val_loader, optimizer, device, False, pos_weight, args.gamma, args.alpha)

#         # Debug prints for epoch 1 to sanity-check
#         if epoch == 1:
#             if train_probs.size:
#                 print(f"\nDebug - Train probs range: [{np.min(train_probs):.4f}, {np.max(train_probs):.4f}]")
#                 print(f"Debug - Train probs mean: {np.mean(train_probs):.4f}")
#                 print(f"Debug - Train probs NaN count: {np.sum(np.isnan(train_probs))}")
#             if val_probs.size:
#                 print(f"Debug - Val probs range: [{np.min(val_probs):.4f}, {np.max(val_probs):.4f}]")
#                 print(f"Debug - Val probs mean: {np.mean(val_probs):.4f}")
#                 print(f"Debug - Val probs NaN count: {np.sum(np.isnan(val_probs))}")
#             print(f"Debug - Train labels distribution: {np.bincount(train_trues.astype(int)) if train_trues.size else 'empty'}")
#             print(f"Debug - Val labels distribution: {np.bincount(val_trues.astype(int)) if val_trues.size else 'empty'}")
#             if val_probs.size:
#                 print(f"Debug - Val predictions at threshold 0.5: {np.bincount((val_probs >= 0.5).astype(int))}")

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#         val_metrics = compute_metrics(val_trues, val_probs)
#         log_line = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_metrics['f1']:.4f}"
#         print(log_line)
#         train_log_lines.append(log_line)

#         # Save best checkpoint (full checkpoint with scaler, cat_maps)
#         if val_metrics["f1"] > best_f1:
#             best_f1 = val_metrics["f1"]
#             best_ckpt_path = os.path.join(args.save_results, f"best_model_ckpt_epoch{epoch}.pt")
#             torch.save({
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "scaler": scaler,
#                 "cat_maps": cat_maps,
#                 "args": vars(args)
#             }, best_ckpt_path)
#             print("Saved best model checkpoint to:", best_ckpt_path)

#     # -------------------------
#     # Load best model before final testing
#     # -------------------------
#     if best_ckpt_path:
#         print("\nLoading best checkpoint for final evaluation...")
#         ckpt = torch.load(best_ckpt_path, map_location=device)
#         model.load_state_dict(ckpt["model_state_dict"])

#     # -------------------------
#     # Final evaluation (test set)
#     # -------------------------
#     _, final_probs, final_trues = run_epoch(model, test_loader, optimizer, device, False, pos_weight, args.gamma, args.alpha)
#     final_metrics = compute_metrics(final_trues, final_probs)

#     # Save training log
#     training_log_path = os.path.join(args.save_results, "training_log.txt")
#     with open(training_log_path, "w") as f:
#         f.write("\n".join(train_log_lines))
#     print("Saved training log to:", training_log_path)

#     # Save final metrics human-readable
#     metrics_txt_path = os.path.join(args.save_results, "metrics.txt")
#     with open(metrics_txt_path, "w") as f:
#         f.write("Final Test Metrics\n=================\n")
#         f.write(f"ROC-AUC : {final_metrics['roc_auc']:.6f}\n")
#         f.write(f"PR-AUC  : {final_metrics['pr_auc']:.6f}\n")
#         f.write(f"F1      : {final_metrics['f1']:.6f}\n")
#         f.write(f"Recall  : {final_metrics['recall']:.6f}\n")
#         f.write(f"Precision: {final_metrics['precision']:.6f}\n")
#         f.write(f"Confusion matrix:\n{final_metrics['confusion_matrix']}\n")
#         f.write(f"\nBest checkpoint: {best_ckpt_path}\n")
#     print("Saved final metrics to:", metrics_txt_path)

#     # Save plots
#     plot_loss(train_losses, val_losses, args.save_results)
#     if final_probs.size:
#         plot_roc(final_trues, final_probs, args.save_results)
#         plot_pr(final_trues, final_probs, args.save_results)
#     plot_confusion(final_metrics["confusion_matrix"], args.save_results)

#     # Save prediction probs
#     preds_path = os.path.join(args.save_results, "pred_probs.npy")
#     np.save(preds_path, final_probs)
#     print("Saved prediction probabilities to:", preds_path)

#     print("\nTraining complete. Results and models saved under:", args.save_results)


# if __name__ == "__main__":
#     main()
import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_loader import load_unsw_kaggle_dataset
from models.logistic_regression import LogisticRegressionModel
from models.xgboost import XGBoostModel
from models.mlp import TabularDataset, MLPTrainer
from models.ft_transformer import FTDataset, FTTransformerTrainer
from utils.metrics import evaluate_binary, compute_curves
from utils.plotting import plot_loss_curves, plot_roc_pr_cm


def main():
    parser = argparse.ArgumentParser(description="Train NIDS models on UNSW_NB15 (Kaggle).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv")
    parser.add_argument("--model", type=str, required=True,
                        choices=["logreg", "xgb", "mlp", "ftt"],
                        help="Which model to train")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs (used for MLP/FTT)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size (used for MLP/FTT)")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Base output directory")
    args = parser.parse_args()

    # Load data
    (X_train_full, X_test_full,
     X_train_num, X_test_num,
     X_train_cat, X_test_cat,
     y_train, y_test,
     num_cols, cat_cols,
     scaler) = load_unsw_kaggle_dataset(args.data_dir)

    # Imbalance ratio
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight_ratio = max(1.0, n_neg / max(1, n_pos))
    print(f"[Info] Train label counts: 0 -> {n_neg}, 1 -> {n_pos}, "
          f"pos_weight_ratio (neg/pos) = {pos_weight_ratio:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] Using device:", device)

    # Create model-specific output dir
    model_name = args.model
    out_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # ===========================
    # Logistic Regression
    # ===========================
    if model_name == "logreg":
        # Single train/val split (for "best" – but only one shot)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        model = LogisticRegressionModel()
        model.fit(X_tr, y_tr)

        # Validation metrics (for reporting)
        val_probs = model.predict_proba(X_val)
        val_metrics = evaluate_binary(y_val, val_probs)
        print("[LogReg] Val metrics:", val_metrics)

        # "Best" model is this single fit → save once
        best_model_path = os.path.join(out_dir, "logreg_best.pkl")
        model.save(best_model_path)
        print(f"[LogReg] Model saved to {best_model_path}")

        # Test metrics
        test_probs = model.predict_proba(X_test_full)
        test_metrics = evaluate_binary(y_test, test_probs)
        print("[LogReg] Test metrics:", test_metrics)

        curves = compute_curves(y_test, test_probs)
        plot_roc_pr_cm(curves, out_dir, title_prefix="LogReg")

        # Save metrics
        metrics_path = os.path.join(out_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("Logistic Regression Metrics\n")
            f.write(f"Val: {val_metrics}\n")
            f.write(f"Test: {test_metrics}\n")
            f.write(f"ROC-AUC: {curves['roc_auc']:.6f}, PR-AUC: {curves['pr_auc']:.6f}\n")
        print(f"[LogReg] Metrics saved to {metrics_path}")
        return

    # ===========================
    # XGBoost
    # ===========================
    if model_name == "xgb":
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        model = XGBoostModel(pos_weight_ratio)
        model.fit(X_tr, y_tr)

        val_probs = model.predict_proba(X_val)
        val_metrics = evaluate_binary(y_val, val_probs)
        print("[XGB] Val metrics:", val_metrics)

        best_model_path = os.path.join(out_dir, "xgboost_best.json")
        model.save(best_model_path)
        print(f"[XGB] Model saved to {best_model_path}")

        test_probs = model.predict_proba(X_test_full)
        test_metrics = evaluate_binary(y_test, test_probs)
        print("[XGB] Test metrics:", test_metrics)

        curves = compute_curves(y_test, test_probs)
        plot_roc_pr_cm(curves, out_dir, title_prefix="XGBoost")

        metrics_path = os.path.join(out_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("XGBoost Metrics\n")
            f.write(f"Val: {val_metrics}\n")
            f.write(f"Test: {test_metrics}\n")
            f.write(f"ROC-AUC: {curves['roc_auc']:.6f}, PR-AUC: {curves['pr_auc']:.6f}\n")
        print(f"[XGB] Metrics saved to {metrics_path}")
        return

    # ===========================
    # MLP (PyTorch)
    # ===========================
    if model_name == "mlp":
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        train_ds = TabularDataset(X_tr, y_tr)
        val_ds = TabularDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        trainer = MLPTrainer(input_dim=X_train_full.shape[1],
                             pos_weight_ratio=pos_weight_ratio,
                             lr=1e-3)
        best_model_path = os.path.join(out_dir, "mlp_best.pt")
        history = trainer.train_epochs(train_loader, val_loader,
                                       epochs=args.epochs,
                                       device=device,
                                       save_path=best_model_path)

        plot_loss_curves(history, out_dir, title_prefix="MLP")

        # Get best validation metrics from training history
        best_val_f1 = max(history["val_f1"]) if history["val_f1"] else 0.0
        best_val_epoch = history["val_f1"].index(best_val_f1) + 1 if history["val_f1"] else 0
        best_val_loss = history["val_loss"][best_val_epoch - 1] if best_val_epoch > 0 else float("inf")
        
        # Recompute validation metrics for the best model
        trainer.load_best(best_model_path, device)
        val_probs = trainer.predict_proba(X_val, device=device, batch_size=args.batch_size)
        val_metrics = evaluate_binary(y_val, val_probs)
        print("[MLP] Val metrics:", val_metrics)

        # Load best model and evaluate on test
        test_probs = trainer.predict_proba(X_test_full, device=device, batch_size=args.batch_size)
        test_metrics = evaluate_binary(y_test, test_probs)
        print("[MLP] Test metrics:", test_metrics)

        curves = compute_curves(y_test, test_probs)
        plot_roc_pr_cm(curves, out_dir, title_prefix="MLP")

        metrics_path = os.path.join(out_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("MLP Metrics\n")
            f.write(f"Val: {val_metrics}\n")
            f.write(f"Test: {test_metrics}\n")
            f.write(f"ROC-AUC: {curves['roc_auc']:.6f}, PR-AUC: {curves['pr_auc']:.6f}\n")
        print(f"[MLP] Metrics saved to {metrics_path}")
        return

    # ===========================
    # FT-Transformer
    # ===========================
    if model_name == "ftt":
        Xn_tr, Xn_val, yc_tr, yc_val, Xc_tr, Xc_val = train_test_split(
            X_train_num, y_train, X_train_cat,
            test_size=0.2, random_state=42, stratify=y_train
        )

        train_ds = FTDataset(Xn_tr, Xc_tr, yc_tr)
        val_ds = FTDataset(Xn_val, Xc_val, yc_val)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        if X_train_cat.shape[1] > 0:
            # Calculate category sizes from BOTH train and test to avoid index out of bounds
            # This ensures all possible categorical values are covered
            all_cat = np.vstack([X_train_cat, X_test_cat]) if X_test_cat.shape[0] > 0 else X_train_cat
            categories = tuple(int(np.max(all_cat[:, i]) + 1) for i in range(X_train_cat.shape[1]))
            print(f"[FTT] Category sizes (from train+test): {categories}")
            
            # Validate and clamp test categorical values to valid ranges
            for i in range(X_test_cat.shape[1]):
                max_val = categories[i] - 1
                if np.any(X_test_cat[:, i] > max_val):
                    n_out_of_bounds = np.sum(X_test_cat[:, i] > max_val)
                    print(f"[FTT] Warning: {n_out_of_bounds} test samples have out-of-bounds values "
                          f"in categorical column {i} (max allowed: {max_val}). Clamping to valid range.")
                    X_test_cat[:, i] = np.clip(X_test_cat[:, i], 0, max_val)
        else:
            categories = (2,)  # dummy
        num_cont = X_train_num.shape[1]

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

        best_model_path = os.path.join(out_dir, "ftt_best.pt")
        history = trainer.train_epochs(train_loader, val_loader,
                                       epochs=args.epochs,
                                       device=device,
                                       save_path=best_model_path)

        plot_loss_curves(history, out_dir, title_prefix="FT-Transformer")

        # Recompute validation metrics for the best model
        trainer.load_best(best_model_path, device)
        val_probs = trainer.predict_proba(Xn_val, Xc_val, device=device, batch_size=args.batch_size)
        val_metrics = evaluate_binary(yc_val, val_probs)
        print("[FTT] Val metrics:", val_metrics)

        # Load best model and evaluate on test
        test_probs = trainer.predict_proba(X_test_num, X_test_cat, device=device, batch_size=args.batch_size)
        test_metrics = evaluate_binary(y_test, test_probs)
        print("[FTT] Test metrics:", test_metrics)

        curves = compute_curves(y_test, test_probs)
        plot_roc_pr_cm(curves, out_dir, title_prefix="FT-Transformer")

        metrics_path = os.path.join(out_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("FT-Transformer Metrics\n")
            f.write(f"Val: {val_metrics}\n")
            f.write(f"Test: {test_metrics}\n")
            f.write(f"ROC-AUC: {curves['roc_auc']:.6f}, PR-AUC: {curves['pr_auc']:.6f}\n")
        print(f"[FTT] Metrics saved to {metrics_path}")
        return


if __name__ == "__main__":
    main()
