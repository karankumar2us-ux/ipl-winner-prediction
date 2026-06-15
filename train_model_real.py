import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('/home/claude/ipl_project/ipl_matches_real.csv')
print("Raw shape:", df.shape)

# Standardize renamed franchises to a single current name
rename_map = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}

for col in ['team1', 'team2', 'toss_winner', 'winner']:
    df[col] = df[col].replace(rename_map)

# Remove rows where winner is Unknown / not a valid team
valid_teams = set(df['team1'].unique()) | set(df['team2'].unique())
df = df[df['winner'].isin(valid_teams)]

# Drop rows where winner isn't team1 or team2 (data inconsistencies)
df = df[(df['winner'] == df['team1']) | (df['winner'] == df['team2'])]

print("Cleaned shape:", df.shape)
print("\nTeams:", sorted(valid_teams))
print("\nWinner distribution:\n", df['winner'].value_counts())

df.to_csv('/home/claude/ipl_project/ipl_matches_clean.csv', index=False)

# ---- Encode and train ----
encoders = {}
df_encoded = df.copy()
for col in ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner']:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df_encoded[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']]
y = df_encoded['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))
print(f"\nLogistic Regression Accuracy: {log_acc:.2%}")

rf_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.2%}")

best_model, best_name, best_acc = (rf_model, "Random Forest", rf_acc) if rf_acc >= log_acc else (log_model, "Logistic Regression", log_acc)
print(f"\nBest Model: {best_name} ({best_acc:.2%})")
print(f"Baseline (predicting most frequent team): {y.value_counts(normalize=True).max():.2%}")

# Feature importance
if best_name == "Random Forest":
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importance:\n", importance)

    plt.figure(figsize=(6,4))
    importance.plot(kind="bar", color="#2E75B6")
    plt.title("Feature Importance - Random Forest")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("/home/claude/ipl_project/feature_importance.png", dpi=150)
    plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, rf_preds)
labels = encoders['winner'].classes_
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Random Forest (Real IPL Data 2008-2025)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("/home/claude/ipl_project/confusion_matrix.png", dpi=150)
plt.close()

joblib.dump(best_model, "/home/claude/ipl_project/ipl_model.pkl")
joblib.dump(encoders, "/home/claude/ipl_project/encoders.pkl")
print("\nModel and encoders saved.")
