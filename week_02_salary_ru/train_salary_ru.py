"""
Train a TextCNN salary predictor on Russian IT vacancies (hh.ru dump).

Adaptation of HSE NLP week-2 seminar pipeline:
GloVe-twitter -> Navec (RU), UK salaries (£) -> RUB salaries.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from navec import Navec
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "hh_it_vacancies.csv"
NAVEC_PATH = ROOT / "artifacts" / "navec_hudlit_v1_12B_500K_300d_100q.tar"
OUT_DIR = ROOT / "artifacts"
PROCESSED_CSV = ROOT / "data" / "vacancies_clean.csv"

MIN_COUNT = 5
HID_SIZE = 64
WINDOW = 3
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
MAX_LEN = 64
SEED = 42
SALARY_MIN = 20_000
SALARY_MAX = 400_000

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_skills(raw) -> str:
    if pd.isna(raw):
        return ""
    try:
        items = ast.literal_eval(str(raw))
        names = [x.get("name", "") for x in items if isinstance(x, dict)]
        return " ".join(names)
    except Exception:
        return re.sub(r"[^\w\s+/.-]", " ", str(raw), flags=re.UNICODE)


def salary_rub(row) -> float | None:
    lo, hi = row["salary_from"], row["salary_to"]
    if pd.isna(lo) and pd.isna(hi):
        return None
    if pd.isna(lo):
        value = float(hi)
    elif pd.isna(hi):
        value = float(lo)
    else:
        value = 0.5 * (float(lo) + float(hi))
    # rough net conversion if marked as gross
    if bool(row.get("gross")) is True:
        value *= 0.87
    return value


def build_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["skills_text"] = df["key_skills"].map(parse_skills)
    df["Title"] = df["name"].fillna("").astype(str)
    df["FullDescription"] = (
        df["Title"] + " " + df["skills_text"] + " " + df["specializations_name"].fillna("").astype(str)
    )
    df["SalaryRub"] = df.apply(salary_rub, axis=1)
    df = df[df["SalaryRub"].notna()].copy()
    df = df[(df["SalaryRub"] >= SALARY_MIN) & (df["SalaryRub"] <= SALARY_MAX)].copy()
    df["Log1pSalary"] = np.log1p(df["SalaryRub"]).astype("float32")

    for col in ["area_name", "experience_name", "schedule_name", "employment_name"]:
        df[col] = df[col].fillna("NaN").astype(str)

    df = df.reset_index(drop=True)
    return df


def tokenize_corpus(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    tokenizer = nltk.tokenize.WordPunctTokenizer()

    def tok(row: str) -> str:
        return " ".join(tokenizer.tokenize(str(row))).lower()

    for col in text_cols:
        df[col] = df[col].map(tok)
    return df


def build_vocab(df: pd.DataFrame, text_cols: list[str], min_count: int):
    counts: Counter = Counter()
    for col in text_cols:
        for text in df[col].values:
            counts.update(text.split())
    tokens = sorted(t for t, c in counts.items() if c >= min_count)
    unk, pad = "UNK", "PAD"
    tokens = [unk, pad] + tokens
    token_to_id = {t: i for i, t in enumerate(tokens)}
    return tokens, token_to_id, unk, pad, counts


def as_matrix(sequences, token_to_id, unk_ix, pad_ix, max_len=None):
    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))
    max_len = min(max(map(len, sequences)), max_len or float("inf"))
    matrix = np.full((len(sequences), max_len), np.int32(pad_ix))
    for i, seq in enumerate(sequences):
        row_ix = [token_to_id.get(w, unk_ix) for w in seq[:max_len]]
        matrix[i, : len(row_ix)] = row_ix
    return matrix


def emb_matrix_from_navec(navec: Navec, tokens: list[str]) -> torch.Tensor:
    dim = int(navec.pq.dim)
    matrix = np.zeros((len(tokens), dim), dtype=np.float32)
    hits = 0
    for i, token in enumerate(tokens):
        vec = navec.get(token)
        if vec is None:
            # try lowercase / stripped punctuation variants already lowercased
            vec = navec.get(token.replace("ё", "е"))
        if vec is not None:
            matrix[i] = vec
            hits += 1
        else:
            matrix[i] = np.random.normal(scale=0.6, size=(dim,)).astype(np.float32)
    print(f"Navec coverage: {hits}/{len(tokens)} ({100 * hits / len(tokens):.1f}%)")
    return torch.tensor(matrix, dtype=torch.float32)


class TextEncoder(nn.Module):
    def __init__(self, embeddings, hid_size, window, dropout_prob=0.2, pooling="mot"):
        super().__init__()
        self.emb = embeddings
        self.emb_size = self.emb.weight.shape[1]
        self.pooling_type = pooling
        self.conv1 = nn.Conv1d(self.emb_size, hid_size, window)
        self.conv2 = nn.Conv1d(self.emb_size, hid_size, window)
        self.bn1 = nn.BatchNorm1d(hid_size)
        self.bn2 = nn.BatchNorm1d(hid_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(2 * hid_size, hid_size)

    def forward(self, text_indices):
        embs = self.emb(text_indices).transpose(2, 1)
        conv_outs1 = self.dropout(self.bn1(self.conv1(embs)))
        conv_outs2 = self.dropout(self.bn2(self.conv2(embs)))
        conv_outs = torch.cat((conv_outs1, conv_outs2), dim=1)
        if self.pooling_type == "mot":
            pooled = torch.max(conv_outs, axis=-1).values
        else:
            weighted = F.softmax(conv_outs, dim=1) * conv_outs
            pooled = torch.max(weighted, axis=-1).values
        return self.fc(pooled)


class SalaryPredictorRU(nn.Module):
    def __init__(self, embeddings, n_cat_features, hid_size=64, dropout_prob=0.2):
        super().__init__()
        self.hid_size = hid_size
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.title_encoder = TextEncoder(self.emb, hid_size, WINDOW, pooling="softmax")
        self.descr_encoder = TextEncoder(self.emb, hid_size, WINDOW, pooling="softmax")
        self.bn_title = nn.BatchNorm1d(hid_size)
        self.bn_descr = nn.BatchNorm1d(hid_size)
        self.dropout_text = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(n_cat_features, hid_size)
        self.bn1 = nn.BatchNorm1d(hid_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hid_size * 3, 1)

    def forward(self, batch):
        title = F.relu(self.title_encoder(batch["Title"]))
        descr = F.relu(self.descr_encoder(batch["FullDescription"]))
        title = self.dropout_text(self.bn_title(title))
        descr = self.dropout_text(self.bn_descr(descr))
        cat = F.relu(self.fc1(batch["Categorical"]))
        cat = self.dropout(self.bn1(cat))
        x = torch.cat((title, descr, cat), dim=-1)
        return self.fc2(x).squeeze(-1)


def make_batch(data, token_to_id, unk_ix, pad_ix, vectorizer, categorical_columns, max_len=None):
    batch = {
        "Title": as_matrix(data["Title"].values, token_to_id, unk_ix, pad_ix, max_len),
        "FullDescription": as_matrix(
            data["FullDescription"].values, token_to_id, unk_ix, pad_ix, max_len
        ),
        "Categorical": vectorizer.transform(data[categorical_columns].apply(dict, axis=1)),
        "Log1pSalary": data["Log1pSalary"].values.astype(np.float32),
        "SalaryRub": data["SalaryRub"].values.astype(np.float32),
    }
    out = {}
    for k, arr in batch.items():
        if k in ("Title", "FullDescription"):
            out[k] = torch.tensor(arr, device=device, dtype=torch.int64)
        else:
            out[k] = torch.tensor(arr, device=device)
    return out


def iterate_minibatches(data, batch_size, token_to_id, unk_ix, pad_ix, vectorizer, cat_cols, shuffle=True):
    indices = np.arange(len(data))
    if shuffle:
        indices = np.random.permutation(indices)
    for start in range(0, len(indices), batch_size):
        yield make_batch(
            data.iloc[indices[start : start + batch_size]],
            token_to_id,
            unk_ix,
            pad_ix,
            vectorizer,
            cat_cols,
            max_len=MAX_LEN,
        )


@torch.no_grad()
def evaluate(model, data, token_to_id, unk_ix, pad_ix, vectorizer, cat_cols):
    model.eval()
    sq = abs_e = n = 0.0
    abs_rub = 0.0
    for batch in iterate_minibatches(
        data, BATCH_SIZE, token_to_id, unk_ix, pad_ix, vectorizer, cat_cols, shuffle=False
    ):
        pred = model(batch)
        sq += torch.sum((pred - batch["Log1pSalary"]) ** 2).item()
        abs_e += torch.sum(torch.abs(pred - batch["Log1pSalary"])).item()
        pred_rub = torch.expm1(pred)
        abs_rub += torch.sum(torch.abs(pred_rub - batch["SalaryRub"])).item()
        n += len(pred)
    return {
        "mse_log": sq / n,
        "mae_log": abs_e / n,
        "mae_rub": abs_rub / n,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("device:", device)

    print("Loading & cleaning CSV...")
    df = build_frame(DATA_CSV)
    text_cols = ["Title", "FullDescription"]
    categorical_columns = ["area_name", "experience_name", "schedule_name", "employment_name"]
    df = tokenize_corpus(df, text_cols)
    df.to_csv(PROCESSED_CSV, index=False)
    print("clean rows:", len(df), "median salary:", float(df["SalaryRub"].median()))

    # company-like rare city collapse not needed; keep top areas as-is
    tokens, token_to_id, UNK, PAD, counts = build_vocab(df, text_cols, MIN_COUNT)
    unk_ix, pad_ix = token_to_id[UNK], token_to_id[PAD]
    print("vocab size:", len(tokens), "unique raw tokens:", len(counts))

    vectorizer = DictVectorizer(dtype=np.float32, sparse=False)
    vectorizer.fit(df[categorical_columns].apply(dict, axis=1))
    print("categorical dims:", len(vectorizer.vocabulary_))

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("Loading Navec...")
    navec = Navec.load(str(NAVEC_PATH))
    emb = emb_matrix_from_navec(navec, tokens)

    model = SalaryPredictorRU(emb, n_cat_features=len(vectorizer.vocabulary_), hid_size=HID_SIZE).to(
        device
    )
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_mae_rub = float("inf")
    best_path = OUT_DIR / "salary_ru_best.pt"

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch in tqdm(
            iterate_minibatches(
                train_df, BATCH_SIZE, token_to_id, unk_ix, pad_ix, vectorizer, categorical_columns
            ),
            total=max(1, len(train_df) // BATCH_SIZE),
            desc=f"epoch {epoch}",
        ):
            pred = model(batch)
            loss = criterion(pred, batch["Log1pSalary"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate(model, val_df, token_to_id, unk_ix, pad_ix, vectorizer, categorical_columns)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(row)
        if metrics["mae_rub"] < best_mae_rub:
            best_mae_rub = metrics["mae_rub"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tokens": tokens,
                    "token_to_id": token_to_id,
                    "vectorizer": vectorizer,
                    "categorical_columns": categorical_columns,
                    "metrics": metrics,
                    "config": {
                        "hid_size": HID_SIZE,
                        "window": WINDOW,
                        "max_len": MAX_LEN,
                        "emb_dim": emb.shape[1],
                    },
                },
                best_path,
            )

    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({"history": history, "best_mae_rub": best_mae_rub}, indent=2), encoding="utf-8")

    # a few qualitative predictions
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state"])
    model.eval()
    demo = val_df.sample(5, random_state=SEED)
    batch = make_batch(demo, token_to_id, unk_ix, pad_ix, vectorizer, categorical_columns, MAX_LEN)
    with torch.no_grad():
        pred = torch.expm1(model(batch)).cpu().numpy()
    examples = []
    for i, (_, row) in enumerate(demo.iterrows()):
        examples.append(
            {
                "title": row["name"] if "name" in row else row["Title"],
                "true_rub": float(row["SalaryRub"]),
                "pred_rub": float(pred[i]),
                "area": row["area_name"],
                "experience": row["experience_name"],
            }
        )
    (OUT_DIR / "demo_predictions.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved:", best_path, metrics_path)
    print("Best MAE (RUB):", round(best_mae_rub))


if __name__ == "__main__":
    main()
