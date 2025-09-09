# Authenti News: End-to-End Fake News Detection (No Web App)
# Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn (optional), kagglehub (optional), pickle

import os
import re
import string
import warnings
warnings.filterwarnings("ignore")

# Data & math
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML & NLP
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import pickle

# -----------------------------------------------------------------------------
# 1) Data acquisition: KaggleHub (optional). Alternatively, set dataset_dir manually
# -----------------------------------------------------------------------------
USE_KAGGLEHUB = True

if USE_KAGGLEHUB:
    try:
        import kagglehub
        dataset_dir = kagglehub.dataset_download("jainpooja/fake-news-detection")
        print(f"[Info] Downloaded dataset to: {dataset_dir}")
    except Exception as e:
        print(f"[Warn] KaggleHub unavailable or failed: {e}")
        print("[Action] Set dataset_dir manually to a local folder containing Fake.csv and True.csv")
        dataset_dir = "./data"  # Fallback; update this to your local dataset path
else:
    dataset_dir = "./data"  # Update this to your local dataset path

fake_csv_path = os.path.join(dataset_dir, "Fake.csv")
true_csv_path = os.path.join(dataset_dir, "True.csv")

if not (os.path.exists(fake_csv_path) and os.path.exists(true_csv_path)):
    raise FileNotFoundError(
        f"Could not find Fake.csv/True.csv in {dataset_dir}. "
        f"Check paths or set USE_KAGGLEHUB accordingly."
    )

# -----------------------------------------------------------------------------
# 2) Load data and label
# -----------------------------------------------------------------------------
df_fake = pd.read_csv(fake_csv_path)
df_true = pd.read_csv(true_csv_path)

print(f"[Info] Fake shape: {df_fake.shape} | True shape: {df_true.shape}")

df_fake["class"] = 0  # fake
df_true["class"] = 1  # real

# -----------------------------------------------------------------------------
# 3) Manual holdout (leakage prevention)
# -----------------------------------------------------------------------------
df_fake_manual = df_fake.tail(10).copy()
df_true_manual = df_true.tail(10).copy()
df_fake = df_fake.iloc[:-10].copy()
df_true = df_true.iloc[:-10].copy()

df_manual = pd.concat([df_fake_manual, df_true_manual], axis=0).sample(frac=1, random_state=42)
df_manual.to_csv("manual_testing.csv", index=False)
print(f"[Info] Manual testing set saved: manual_testing.csv | shape={df_manual.shape}")

# -----------------------------------------------------------------------------
# 4) Merge, prune, shuffle
# -----------------------------------------------------------------------------
df = pd.concat([df_fake, df_true], axis=0)
drop_cols = [c for c in ["title", "subject", "date"] if c in df.columns]
df = df.drop(columns=drop_cols, errors="ignore")

missing = int(df.isnull().sum().sum())
print(f"[Info] Missing values in merged data: {missing}")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 5) Text preprocessing
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("[Info] Cleaning text...")
df["text"] = df["text"].apply(clean_text)
print("[Info] Text cleaning complete.")

# -----------------------------------------------------------------------------
# 6) Features & split
# -----------------------------------------------------------------------------
X = df["text"].values
y = df["class"].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"[Info] Split: train={len(X_train_text)} | test={len(X_test_text)}")

# -----------------------------------------------------------------------------
# 7) TF-IDF vectorization
# -----------------------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)
print("[Info] Vectorization complete. Shape:", X_train.shape, X_test.shape)

# -----------------------------------------------------------------------------
# 8) Train models
# -----------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

preds = {}
probs = {}
accuracies = {}

print("[Info] Training models...")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    preds[name] = y_pred
    accuracies[name] = accuracy_score(y_test, y_pred)
    # Probabilities if available
    if hasattr(clf, "predict_proba"):
        probs[name] = clf.predict_proba(X_test)[:, 1]
    else:
        probs[name] = None
    print(f"  - {name}: accuracy={accuracies[name]:.4f}")

# -----------------------------------------------------------------------------
# 9) Evaluation: classification report & confusion matrices
# -----------------------------------------------------------------------------
def print_reports():
    print("\n=== Classification Reports ===")
    for name, y_pred in preds.items():
        print(f"\n[{name}]")
        print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
    print("\n=== Confusion Matrices ===")
    for name, y_pred in preds.items():
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n[{name}]\n{cm}")

print_reports()

# -----------------------------------------------------------------------------
# 10) ROC curves and AUC
# -----------------------------------------------------------------------------
roc_data = {}
for name, p in probs.items():
    if p is not None:
        fpr, tpr, _ = roc_curve(y_test, p)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)

# -----------------------------------------------------------------------------
# 11) Performance dashboard (matplotlib)
# -----------------------------------------------------------------------------
def plot_dashboard():
    plt.figure(figsize=(14, 10))

    # (1) Accuracy bar chart
    plt.subplot(2, 2, 1)
    mnames = list(accuracies.keys())
    accvals = [accuracies[m] for m in mnames]
    bars = plt.bar(mnames, accvals, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=20)
    for b, v in zip(bars, accvals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # (2) ROC Curves
    plt.subplot(2, 2, 2)
    if len(roc_data) > 0:
        for name, (fpr, tpr, roc_auc) in roc_data.items():
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No probability outputs available", ha="center")

    # (3) Confusion matrix for top model (by accuracy)
    plt.subplot(2, 2, 3)
    best_model_name = max(accuracies, key=accuracies.get)
    cm_best = confusion_matrix(y_test, preds[best_model_name])
    sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # (4) Summary table
    plt.subplot(2, 2, 4)
    plt.axis("off")
    table_data = [[m, f"{accuracies[m]:.4f}"] for m in mnames]
    table = plt.table(cellText=table_data, colLabels=["Model", "Accuracy"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.4)
    plt.title("Performance Summary")

    plt.tight_layout()
    outpath = "model_performance_dashboard.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Info] Dashboard saved: {outpath}")

plot_dashboard()

# -----------------------------------------------------------------------------
# 12) Persistence: save models and vectorizer
# -----------------------------------------------------------------------------
print("[Info] Saving models and vectorizer...")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

for name, clf in models.items():
    filename = f"{name.lower().replace(' ', '_')}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(clf, f)
        print(f"  - Saved: {filename}")

# -----------------------------------------------------------------------------
# 13) Manual testing utility (no web)
# -----------------------------------------------------------------------------
LABEL_MAP = {0: "Fake", 1: "Real"}

def manual_testing(article_text: str, show_prob: bool = True):
    """Run all models on a single article, print per-model predictions and majority vote."""
    cleaned = clean_text(article_text)
    X_vec = tfidf.transform([cleaned])

    votes = []
    print("\n=== Manual Test ===")
    print(f"Snippet: {article_text[:160].strip()}{'...' if len(article_text)>160 else ''}\n")

    for name, clf in models.items():
        pred = int(clf.predict(X_vec)[0])
        votes.append(pred)
        if hasattr(clf, "predict_proba") and show_prob:
            p = float(clf.predict_proba(X_vec)[0, 1])  # prob of Real (class=1)
            print(f"{name:20s} -> {LABEL_MAP[pred]}  | P(Real)={p:.3f}")
        else:
            print(f"{name:20s} -> {LABEL_MAP[pred]}")

    fake_votes = sum(1 for v in votes if v == 0)
    real_votes = sum(1 for v in votes if v == 1)
    final = "Fake" if fake_votes > real_votes else "Real"
    print(f"\nMajority Vote: {final}  (Fake={fake_votes}, Real={real_votes})")
    return final, {"fake": fake_votes, "real": real_votes}

# -----------------------------------------------------------------------------
# 14) Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample_news = """
    Scientists have discovered a new planet beyond Pluto, three times the size of Earth.
    Multiple international space agencies confirmed the finding after months of observation.
    """
    manual_testing(sample_news)
    print("\n[Done] Authenti News pipeline execution complete.")
