# Logistic Regression
python train.py --data_dir "D:/CSI4460/Datasets/UNSW_NB15" --model logreg

# XGBoost
python train.py --data_dir "D:/CSI4460/Datasets/UNSW_NB15" --model xgb

# MLP - 100 epochs
python train.py --data_dir "D:/CSI4460/Datasets/UNSW_NB15" --model mlp --epochs 100 --batch_size 512

# FT-Transformer - 100 epochs
python train.py --data_dir "D:/CSI4460/Datasets/UNSW" --out_dir "D:/CSI4460/results"   --model ftt --epochs 100 --batch_size 512
