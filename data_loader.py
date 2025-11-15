import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_unsw_kaggle_dataset(data_dir: str):
    """
    Load UNSW_NB15 Kaggle version:
      - UNSW_NB15_training-set.csv
      - UNSW_NB15_testing-set.csv

    Returns:
      X_train_full, X_test_full: np.array for classical / MLP (num_scaled + cats as float)
      X_train_num, X_test_num: scaled numeric features (for FT-Transformer)
      X_train_cat, X_test_cat: encoded categorical features (for FT-Transformer)
      y_train, y_test: labels as np.array
      num_cols, cat_cols: column names
      scaler: fitted StandardScaler on numeric cols
    """
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    print(f"[Data] Loading train: {train_path}")
    print(f"[Data] Loading test : {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop id if exists
    for df in [train_df, test_df]:
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    # Categorical columns (typical)
    cat_cols = [c for c in ["proto", "service", "state"] if c in train_df.columns]

    # LabelEncode categorical features using combined train+test
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le

    # Numeric columns: all except label, attack_cat, and categorical
    drop_cols = ["label", "attack_cat"]
    num_cols = [c for c in train_df.columns if c not in cat_cols + drop_cols]

    # Labels
    y_train = train_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    # Scale numeric columns
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[num_cols])
    X_test_num = scaler.transform(test_df[num_cols])

    # Categorical as int64
    if cat_cols:
        X_train_cat = train_df[cat_cols].values.astype(np.int64)
        X_test_cat = test_df[cat_cols].values.astype(np.int64)
    else:
        X_train_cat = np.zeros((len(train_df), 0), dtype=np.int64)
        X_test_cat = np.zeros((len(test_df), 0), dtype=np.int64)

    # Full features for classical / MLP: concat numeric + categorical
    if cat_cols:
        X_train_full = np.hstack([X_train_num, X_train_cat.astype(np.float32)])
        X_test_full = np.hstack([X_test_num, X_test_cat.astype(np.float32)])
    else:
        X_train_full = X_train_num
        X_test_full = X_test_num

    print(f"[Data] Train full shape: {X_train_full.shape}, Test full shape: {X_test_full.shape}")
    print(f"[Data] Numeric cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}")

    return (X_train_full, X_test_full,
            X_train_num, X_test_num,
            X_train_cat, X_test_cat,
            y_train, y_test,
            num_cols, cat_cols,
            scaler)
