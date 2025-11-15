import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tab_transformer_pytorch import FTTransformer
from utils.metrics import evaluate_binary_from_logits


class FTDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        if X_cat.shape[1] > 0:
            self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        else:
            self.X_cat = torch.zeros((len(X_num), 0), dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "num": self.X_num[idx],
            "cat": self.X_cat[idx],
            "y": self.y[idx]
        }


class FTTransformerTrainer:
    def __init__(self, categories, num_cont: int, pos_weight_ratio: float,
                 dim=128, depth=4, heads=8, dropout=0.1, lr=1e-4):
        self.categories = categories  # Store category sizes for validation
        self.model = FTTransformer(
            categories=categories,
            num_continuous=num_cont,
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            dim_out=1
        )
        pos_weight_tensor = torch.tensor(pos_weight_ratio, dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

    def train_epochs(self, train_loader, val_loader, epochs: int, device, save_path: str):
        self.model.to(device)
        self.loss_fn.to(device)

        best_f1 = -1.0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []

            for batch in train_loader:
                x_num = batch["num"].to(device)
                x_cat = batch["cat"].to(device) if batch["cat"].numel() > 0 else None
                y = batch["y"].to(device)

                logits = self.model(x_cat, x_num).squeeze(1)
                loss = self.loss_fn(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses)) if train_losses else float("inf")

            # Validation
            self.model.eval()
            val_losses = []
            all_logits = []
            all_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    x_num = batch["num"].to(device)
                    x_cat = batch["cat"].to(device) if batch["cat"].numel() > 0 else None
                    y = batch["y"].to(device)

                    logits = self.model(x_cat, x_num).squeeze(1)
                    loss = self.loss_fn(logits, y)
                    val_losses.append(loss.item())

                    all_logits.append(logits.cpu())
                    all_targets.append(y.cpu())

            avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
            all_logits = torch.cat(all_logits).numpy()
            all_targets = torch.cat(all_targets).numpy()
            val_metrics = evaluate_binary_from_logits(all_targets, all_logits)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            history["val_f1"].append(val_metrics["f1"])

            print(f"[FTT] Epoch {epoch}/{epochs}: "
                  f"train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, val_f1={val_metrics['f1']:.4f}")

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1": best_f1
                }

        if best_state is not None:
            torch.save(best_state, save_path)
            print(f"[FTT] Best model saved to {save_path} with F1={best_f1:.4f}")

        return history

    def load_best(self, path: str, device):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    def predict_proba(self, X_num: np.ndarray, X_cat: np.ndarray, device, batch_size: int = 1024):
        self.model.eval()
        self.model.to(device)

        # Validate and clamp categorical values to prevent index out of bounds
        if X_cat.shape[1] > 0 and len(self.categories) > 0:
            X_cat = X_cat.copy()  # Avoid modifying original array
            for i in range(X_cat.shape[1]):
                max_val = self.categories[i] - 1  # Category size is max_index + 1
                if np.any(X_cat[:, i] > max_val) or np.any(X_cat[:, i] < 0):
                    n_out_of_bounds = np.sum((X_cat[:, i] > max_val) | (X_cat[:, i] < 0))
                    print(f"[FTT] Warning: {n_out_of_bounds} samples have out-of-bounds categorical values "
                          f"in column {i} (valid range: 0-{max_val}). Clamping to valid range.")
                    X_cat[:, i] = np.clip(X_cat[:, i], 0, max_val)

        ds = FTDataset(X_num, X_cat, np.zeros(len(X_num)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in dl:
                x_num = batch["num"].to(device)
                x_cat = batch["cat"].to(device) if batch["cat"].numel() > 0 else None
                logits = self.model(x_cat, x_num).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs)
