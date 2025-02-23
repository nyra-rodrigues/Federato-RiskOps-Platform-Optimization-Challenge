import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report
from lightgbm import LGBMClassifier, log_evaluation
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.encoding import CountFrequencyEncoder

# ==============================================================================
# 1. Data Preparation
# ==============================================================================
def prepare_data(df):
    # Filter rare classes (keep classes with ≥5 samples)
    class_counts = df['y_best'].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df['y_best'].isin(valid_classes)].copy()
    
    # Encode target
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['y_best'])
    
    return df, le

# ==============================================================================
# 2. Feature Engineering Pipeline
# ==============================================================================
class CyclicalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.assign(
            hour_sin=np.sin(2 * np.pi * X['hour_of_day'] / 24),
            hour_cos=np.cos(2 * np.pi * X['hour_of_day'] / 24),
            day_sin=np.sin(2 * np.pi * X['day_of_week'] / 7),
            day_cos=np.cos(2 * np.pi * X['day_of_week'] / 7)
        )

def create_pipeline(cat_features):
    return Pipeline([
        ('num_imputer', ArbitraryNumberImputer(
            variables=['session_length'],
            arbitrary_number=0
        )),
        ('cat_imputer', CategoricalImputer(
            variables=cat_features,
            fill_value='missing'
        )),
        ('cat_encoder', CountFrequencyEncoder(
            variables=cat_features,
            encoding_method='frequency',
            missing_values='ignore'
        )),
        ('cyclical', CyclicalTransformer())
    ])

# ==============================================================================
# 3. Model Training
# ==============================================================================
# ==============================================================================
# 3. Model Training (Simplified Version)
# ==============================================================================
def train_model(df, le):
    # Define features
    num_features = ['session_length', 'time_since_last_session', 
                   'total_past_sessions', 'hour_of_day', 'day_of_week']
    cat_features = ['event_type', 'event_category', 'event_slug']
    
    X = df[num_features + cat_features]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = create_pipeline(cat_features)
    
    # Preprocess data
    X_train_trans = pipeline.fit_transform(X_train, y_train)
    X_test_trans = pipeline.transform(X_test)
    
    # Train model with simplified parameters
    model = LGBMClassifier(
        objective='multiclass',
        num_classes=len(le.classes_),
        n_estimators=200,  # Reduced from 1000
        learning_rate=0.1,  # Increased from 0.05
        max_depth=5,        # Reduced from 7
        n_jobs=-1,
        verbosity=-1
    )
    
    # Simplified fit without validation
    model.fit(X_train_trans, y_train)
    
    return model, pipeline

# ==============================================================================
# 4. Main Execution
# ==============================================================================
if __name__ == "__main__":
    df = pd.read_csv("processed_dataset.csv")
    df_filtered, le = prepare_data(df)
    model, pipeline = train_model(df_filtered, le)
    
    # Save artifacts
    joblib.dump({
        'model': model,
        'pipeline': pipeline,
        'encoder': le
    }, "production_model.pkl")