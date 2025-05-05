import numpy as np
from src.shared.json_tools import load_json_long
from paths import DATA_DIR
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torch
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sklearn.metrics import accuracy_score

def prep_data():
    np.random.seed(42)  # for reproducibility
    X = []
    y = []

    for file in (DATA_DIR / "selected").glob("*.json"):
        if len(X) >= 1000:
            break
        data = load_json_long(file)
        targets = [int(i["target"] >= 0.5) for i in data]
        features = [i["content"] for i in data]

        features = np.array(features)
        targets = np.array(targets)

        pos_indices = np.where(targets == 1)[0]
        neg_indices = np.where(targets == 0)[0]

        min_class_count = min(len(pos_indices), len(neg_indices))
        pos_sample = np.random.choice(pos_indices, min_class_count, replace=False)
        neg_sample = np.random.choice(neg_indices, min_class_count, replace=False)

        balanced_indices = np.concatenate([pos_sample, neg_sample])
        np.random.shuffle(balanced_indices)

        features_balanced = features[balanced_indices]
        y_balanced = targets[balanced_indices]

        X = X + features_balanced.tolist()
        y = y + y_balanced.tolist()
    return X, y

X, y = prep_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train positive samples: {sum(y_train)}, Test positive samples: {sum(y_test)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Conversion to GPU tensors (not used directly by XGBoost)
X_train_gpu = torch.from_numpy(np.array(X_train)).to(device)
y_train_gpu = torch.from_numpy(np.array(y_train)).to(device)
X_test_gpu = torch.from_numpy(np.array(X_test)).to(device)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8],
    "gamma": [0, 0.1],
    "reg_alpha": [0, 1],
    "reg_lambda": [1, 3],
    "min_child_weight": [1, 5]
}


# Generate all combinations of hyperparameters
keys = param_grid.keys()
values = param_grid.values()
all_combinations = [dict(zip(keys, v)) for v in product(*values)]

best_score = 0
best_params = None
results = []

for param in all_combinations:
    model = XGBClassifier(
        device="cuda",
        eval_metric="logloss",
        random_state=42,
        **param
    )

    model.fit(X_train_gpu, y_train_gpu)
    preds = model.predict(X_test_gpu)
    acc = accuracy_score(y_test, preds)

    result = {
        "params": param,
        "accuracy": acc
    }
    results.append(result)

    if acc > best_score:
        print(f"Better score: {acc} - {param}")
        best_score = acc
        best_params = param

print("\nBest Parameters:")
print(best_params)
print(f"Best Accuracy: {best_score:.4f}")