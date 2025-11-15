# Logistic Regression
python train.py --data_dir "D:/CSI4460/Datasets/UNSW" --model logreg

# XGBoost
python train.py --data_dir "D:/CSI4460/Datasets/UNSW" --model xgb

# MLP - 100 epochs
python train.py --data_dir  "D:/CSI4460/Datasets/UNSW" --out_dir "D:/CSI4460/results" --model mlp --epochs 100 --batch_size 512

# FT-Transformer - 100 epochs
python train.py --data_dir "D:/CSI4460/Datasets/UNSW" --out_dir "D:/CSI4460/results"   --model ftt --epochs 100 --batch_size 512
