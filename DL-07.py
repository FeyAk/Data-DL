from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# --- Load dataset ---
bank = fetch_ucirepo(id=222)   # Bank Marketing
X = bank.data.features
y = bank.data.targets
y = y.values.ravel()

# --- Convert categorical features to numeric ---
X = pd.get_dummies(X)   # very important!

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train model ---
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# --- Predict ---
y_pred = clf.predict(X_test)

# --- Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
