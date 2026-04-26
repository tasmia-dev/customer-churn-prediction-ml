# ============================================
# Customer Churn Prediction
# Codec Technologies - Industrial Project
# ============================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # ✅ IMPORTANT FIX
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, recall_score,
                             roc_auc_score, classification_report,
                             confusion_matrix)

import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 50)

# ── 1. LOAD DATA ──────────────────────────────
print("\n📂 Loading dataset...")
df = pd.read_csv('data/telco_churn.csv')
print(f"Shape: {df.shape}")
print(f"Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")

# ── 2. EDA ────────────────────────────────────
print("\n📊 Running EDA...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df['Churn'].value_counts().plot(kind='bar', ax=axes[0,0], color=['steelblue','tomato'])
axes[0,0].set_title('Churn Distribution')

sns.countplot(data=df, x='Contract', hue='Churn', ax=axes[0,1], palette='Set2')
axes[0,1].set_title('Churn by Contract Type')

sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1,0], palette='Set1')
axes[1,0].set_title('Monthly Charges vs Churn')

sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[1,1], palette='Set3')
axes[1,1].set_title('Tenure vs Churn')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150)
print("✅ EDA plots saved as eda_plots.png")

# ── 3. PREPROCESSING ──────────────────────────
print("\n⚙️  Preprocessing data...")

# Drop ID
df.drop('customerID', axis=1, inplace=True)

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("✅ Encoding done!")

# ── 4. SPLIT & SCALE ──────────────────────────
print("\n✂️  Splitting data...")

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Handle any remaining NaN
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")

# ── 5. TRAIN MODELS ───────────────────────────
print("\n🤖 Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }

# ── 6. RESULTS ────────────────────────────────
print("\n" + "=" * 50)
print("   MODEL RESULTS")
print("=" * 50)

results_df = pd.DataFrame(results).T
print(results_df.round(4))

best = results_df['ROC-AUC'].idxmax()
print(f"\n🏆 Best Model: {best} (ROC-AUC: {results_df.loc[best,'ROC-AUC']:.4f})")

# ── 7. ROC CURVE ──────────────────────────────
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, name=name)

ax.set_title('ROC Curves - Model Comparison')
plt.savefig('roc_curves.png', dpi=150)

print("✅ ROC curves saved as roc_curves.png")

# ── 8. FEATURE IMPORTANCE ─────────────────────
feat_importance = pd.Series(
    models['Random Forest'].feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
feat_importance.plot(kind='bar', color='steelblue')
plt.title('Top 10 Important Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)

print("✅ Feature importance saved as feature_importance.png")

print("\n" + "=" * 50)
print("   ✅ PROJECT COMPLETE!")
print("=" * 50)