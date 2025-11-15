## 👉 Folder Structure
```
CSI4460/
│
├── Datasets/UNSW                       # All ML / DL models
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
├── data_loader.py                # Load & preprocess UNSW-NB15
│
├── models/                       # All ML / DL models
│   ├── ft_transformer.py
│   ├── tabnet.py
│   ├── mlp.py
│   ├── logistic_regression.py
│
├── results/                      # Saved outputs & checkpoints
│   ├── ftt/
│   ├── mlp/
│   ├── logistic/
│   ├── tabnet/
│
├── train.py                      # Universal training loop (epochs, validation)
├── utils.py                      # Metrics, plots, imbalance handling
├── config.yaml                   # Experiment configuration
└── main.py                       # Entry point to run training/evaluation
```
## Dataset Download
👉  [**Download the dataset from Kaggle (Website)**](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data)
## Trained Model
👉  [**Download the trained model (Google Drive)**](https://drive.google.com/drive/folders/1Jcmix6MMokSTROgOl4w5VfRs67p8IkSN?usp=drive_link)

