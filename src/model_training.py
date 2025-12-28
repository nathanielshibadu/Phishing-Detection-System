import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Define directories relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Load features and labels
features_path = os.path.join(DATA_DIR, 'features.csv')
labels_path = os.path.join(DATA_DIR, 'labels.csv')

X = pd.read_csv(features_path)
y = pd.read_csv(labels_path).squeeze()  # Ensure y is a Series

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Save model
model_path = os.path.join(MODELS_DIR, 'rf_model.joblib')
joblib.dump(clf, model_path)

# ...existing code...

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score

def train_random_forest(X, y):
    # Split data (stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Parameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10]
    }

    clf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=10,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)

    # Cross-validation score on train set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(search.best_estimator_, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Save best model
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    joblib.dump(search.best_estimator_, model_path)

    return search.best_estimator_

# ...existing code...

# ...existing code...

from xgboost import XGBClassifier

def train_xgboost(X, y):
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_dist = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200, 300]
    }

    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=10,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    # Save model as models/xgboost.json
    model_path = os.path.join(MODELS_DIR, 'xgboost.json')
    best_model.save_model(model_path)

    return best_model

# ...existing code...

# ...existing code...

from sklearn.svm import SVC

def train_svm(X, y):
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_dist = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }

    clf = SVC(probability=True, random_state=42)
    search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=6,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    # Save model as models/svm.pkl
    model_path = os.path.join(MODELS_DIR, 'svm.pkl')
    joblib.dump(best_model, model_path)

    return best_model

# ...existing code...

# ...existing code...

def train_compressed_nn(X, y):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build simple Sequential model
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Quantization (TensorFlow Lite)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        # Save TFLite model
        with open(os.path.join(MODELS_DIR, 'compressed_nn.tflite'), 'wb') as f:
            f.write(tflite_model)
    except Exception as e:
        print(f"TFLite conversion failed: {e}")

    # Save Keras model
    model.save(os.path.join(MODELS_DIR, 'compressed_nn.h5'))

    return model

# ...existing code...

# ...existing code...

def main():
    # Load features and labels
    features_path = os.path.join(DATA_DIR, 'features.csv')
    labels_path = os.path.join(DATA_DIR, 'labels.csv')
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()

    print("Training Random Forest...")
    rf_model = train_random_forest(X, y)
    print("Random Forest trained and saved as models/random_forest.pkl")

    print("Training XGBoost...")
    xgb_model = train_xgboost(X, y)
    print("XGBoost trained and saved as models/xgboost.json")

    print("Training SVM...")
    svm_model = train_svm(X, y)
    print("SVM trained and saved as models/svm.pkl")

    print("Training Compressed Neural Network...")
    nn_model = train_compressed_nn(X, y)
    print("Compressed Neural Network trained and saved as models/compressed_nn.h5 (and .tflite if possible)")

if __name__ == "__main__":
    main()