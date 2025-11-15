import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MLPNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class TabularDataset(Dataset):
    def __init__(self, X_full: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X_full, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx]
        }


class MLPTrainer:
    def __init__(self, input_dim: int, pos_weight_ratio: float, lr=1e-3):
        self.model = MLPNet(input_dim)
        pos_weight_tensor = torch.tensor(pos_weight_ratio, dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_epochs(self, train_loader, val_loader, epochs: int, device, save_path: str):
        self.model.to(device)
        self.loss_fn.to(device)

        best_f1 = -1.0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        from utils.metrics import evaluate_binary_from_logits

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []

            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)

                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
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
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    logits = self.model(x)
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

            print(f"[MLP] Epoch {epoch}/{epochs}: "
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
            print(f"[MLP] Best model saved to {save_path} with F1={best_f1:.4f}")

        return history

    def load_best(self, path: str, device):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    def predict_proba(self, X_full: np.ndarray, device, batch_size: int = 1024):
        self.model.eval()
        self.model.to(device)

        ds = torch.utils.data.TensorDataset(torch.tensor(X_full, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (x,) in dl:
                x = x.to(device)
                logits = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs)
