import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("processed_dataset.csv")

categorical_features = ['event_type', 'event_category', 'event_slug', 'event_display_name']

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  

# 1️⃣ First, Remove Classes with Less than 2 Occurrences (without encoding)
y_counts = df['y_best'].value_counts()
valid_classes = y_counts[y_counts > 1].index  
df = df[df['y_best'].isin(valid_classes)]

# 3️⃣ Sample Data to Reduce Memory Usage
df_sample = df.sample(frac=0.5, random_state=42)

# 4️⃣ Filter AGAIN after sampling
y_sample = df_sample['y_best'].copy()
valid_classes_sample = y_sample.value_counts()[y_sample.value_counts() > 1].index
df_sample = df_sample[df_sample['y_best'].isin(valid_classes_sample)]

# 2️⃣ NOW Encode y_best AFTER all filtering
y_encoder = LabelEncoder()
df_sample['y_best'] = y_encoder.fit_transform(df_sample['y_best'].astype(str))  # <--- KEY CHANGE

# Reset index to avoid issues
df_sample = df_sample.reset_index(drop=True)

# Proceed with X and y
X = df_sample.drop(columns=['y_best'])
y = df_sample['y_best']

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# 4️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5️⃣ Train XGBoost Model
clf = XGBClassifier(n_estimators=50, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
clf.fit(X_train, y_train)

# 6️⃣ Evaluate Performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Save Model & Encoders
joblib.dump(clf, "next_action_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(y_encoder, "y_encoder.pkl")

print("\nModel saved as `next_action_model.pkl`")
print("Label Encoders saved as `label_encoders.pkl` & `y_encoder.pkl`")
