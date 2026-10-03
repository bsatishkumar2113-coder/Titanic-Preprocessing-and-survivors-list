import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

IN_PATH = "/mnt/user-data/uploads/Titanic-Dataset.csv"
OUT_DIR = "/mnt/user-data/outputs"

# ---------------------------------------------------------------
# 1. Import dataset and explore basic info
# ---------------------------------------------------------------
df = pd.read_csv(IN_PATH)
print("Shape:", df.shape)
print("\nFirst rows:\n", df.head())
print("\nData types / non-null counts:")
df.info()
print("\nMissing values per column:\n", df.isnull().sum())
print("\nMissing %:\n", (df.isnull().mean() * 100).round(1))
print("\nSummary statistics:\n", df.describe())

# ---------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------
# Age (numeric, skewed)      -> median imputation
# Embarked (categorical, 2)  -> mode imputation
# Cabin (77% missing)        -> too sparse to impute; keep a Has_Cabin flag, drop the column
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Has_Cabin"] = df["Cabin"].notna().astype(int)

# Drop identifier / free-text columns that carry no direct modelling signal
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
print("\nMissing values after imputation:\n", df.isnull().sum())

# ---------------------------------------------------------------
# 3. Encode categorical features
# ---------------------------------------------------------------
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})          # binary -> label encoding
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)      # nominal -> one-hot encoding
print("\nColumns after encoding:", list(df.columns))

# ---------------------------------------------------------------
# 4. Standardize numerical features (z-score)
# ---------------------------------------------------------------
num_cols = ["Age", "Fare", "SibSp", "Parch"]
df[num_cols] = StandardScaler().fit_transform(df[num_cols])
print("\nScaled columns (mean ~0, std ~1):\n", df[num_cols].describe().loc[["mean", "std"]].round(3))

# ---------------------------------------------------------------
# 5. Visualize outliers with boxplots, then remove them (IQR rule)
# ---------------------------------------------------------------
def boxplots(data, title, path):
    fig, axes = plt.subplots(1, len(num_cols), figsize=(14, 5))
    for ax, col in zip(axes, num_cols):
        ax.boxplot(data[col], patch_artist=True, boxprops=dict(facecolor="#9ecae1"))
        ax.set_title(col)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

boxplots(df, "Boxplots BEFORE outlier removal (standardized)", f"{OUT_DIR}/boxplots_before.png")

# IQR rule on continuous columns only. SibSp/Parch are small-integer counts, so
# the IQR rule would wrongly discard every passenger travelling with family.
iqr_cols = ["Age", "Fare"]
mask = pd.Series(True, index=df.index)
for col in iqr_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    print(f"{col}: bounds [{low:.2f}, {high:.2f}] -> {n_out} outliers")
    mask &= df[col].between(low, high)

df_clean = df[mask].reset_index(drop=True)
print(f"\nRows before: {len(df)} | after: {len(df_clean)} | removed: {len(df) - len(df_clean)}")

boxplots(df_clean, "Boxplots AFTER outlier removal (standardized)", f"{OUT_DIR}/boxplots_after.png")

df_clean.to_csv(f"{OUT_DIR}/titanic_cleaned.csv", index=False)
print("\nSaved cleaned dataset. Final shape:", df_clean.shape)
