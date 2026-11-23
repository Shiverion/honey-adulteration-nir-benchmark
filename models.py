"""The seven classifiers and their fixed, equal-size search spaces."""

import pickle

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import nn
from xgboost import XGBClassifier


CLASSICAL = ("PLS-DA", "SVM", "Random Forest", "XGBoost")
NEURAL = ("MLP", "1D-CNN", "Spectral Transformer")
MODELS = CLASSICAL + NEURAL
SEARCH = {
    "PLS-DA": [{"n_components": n} for n in (2, 4, 8, 12)],
    "SVM": [
        {"kernel": "linear", "C": 0.1}, {"kernel": "linear", "C": 1.0},
        {"kernel": "rbf", "C": 1.0}, {"kernel": "rbf", "C": 10.0},
    ],
    "Random Forest": [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth in (5, None) for leaf in (1, 4)
    ],
    "XGBoost": [
        {"max_depth": depth, "learning_rate": rate}
        for depth in (2, 4) for rate in (0.04, 0.12)
    ],
}
for name in NEURAL:
    SEARCH[name] = [
        {"learning_rate": rate, "weight_decay": decay}
        for rate in (0.001, 0.003) for decay in (0.0001, 0.01)
    ]


class SpectralTransformer(nn.Module):
    def __init__(self, bands):
        super().__init__()
        width, patch_size = 48, 4
        self.patch = nn.Conv1d(1, width, kernel_size=patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.empty(1, bands // patch_size, width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=96, dropout=0.10,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.encoder.layers[1].load_state_dict(nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=96, dropout=0.10,
            batch_first=True, norm_first=True, activation="gelu",
        ).state_dict())
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x):
        tokens = self.patch(x.unsqueeze(1)).transpose(1, 2) + self.position
        return self.head(self.encoder(tokens).mean(dim=1))


def network(name, bands):
    if name == "MLP":
        return nn.Sequential(
            nn.Linear(bands, 32), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1),
        )
    if name == "1D-CNN":
        return nn.Sequential(
            nn.Unflatten(1, (1, bands)), nn.Conv1d(1, 16, 7, padding=3), nn.GELU(),
            nn.AvgPool1d(2), nn.Conv1d(16, 32, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(6), nn.Flatten(), nn.Dropout(0.10), nn.Linear(192, 1),
        )
    return SpectralTransformer(bands)


def fit_model(name, params, x, y, seed):
    positive_weight = np.count_nonzero(y == 0) / np.count_nonzero(y == 1)
    if name in NEURAL:
        torch.manual_seed(seed)
        model = network(name, x.shape[1])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y.astype(np.float32))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"],
        )
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32))
        for _ in range(45):
            for batch in torch.randperm(len(x)).split(64):
                optimizer.zero_grad(set_to_none=True)
                loss(model(x[batch]).ravel(), y[batch]).backward()
                optimizer.step()
        return model.eval()
    if name == "PLS-DA":
        model = PLSRegression(scale=False, **params)
    elif name == "SVM":
        model = SVC(class_weight="balanced", **params)
    elif name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=150, class_weight="balanced", n_jobs=1, random_state=seed, **params,
        )
    elif name == "XGBoost":
        model = XGBClassifier(
            n_estimators=150, tree_method="hist", n_jobs=1, random_state=seed,
            scale_pos_weight=positive_weight, **params,
        )
    else:
        raise ValueError(f"Unknown model {name}")
    return model.fit(x, y)


def decision_score(name, model, x):
    if name in NEURAL:
        with torch.inference_mode():
            return model(torch.from_numpy(x)).ravel().numpy()
    if name == "PLS-DA":
        return model.predict(x).ravel() - 0.5
    if name == "SVM":
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1] - 0.5


def footprint(name, model):
    if name in NEURAL:
        parameters = sum(values.numel() for values in model.parameters())
        snapshot = {key: value.numpy() for key, value in model.state_dict().items()}
    else:
        parameters = None
        snapshot = model
    return len(pickle.dumps(snapshot, protocol=5)), parameters
