import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURE_DIR = "extracted_features"
MODEL_SAVE_PATH = "fall_detection_rf.pkl"

def load_data():
    X, y = [], []
    classes = {'Falling': 0, 'Normal': 1}
    for f in os.listdir(FEATURE_DIR):
        if f.endswith('.npy'):
            label_str = f.split('_')[0]
            if label_str in classes:
                X.append(np.load(os.path.join(FEATURE_DIR, f)))
                y.append(classes[label_str])
    return np.array(X), np.array(y), classes

if __name__ == "__main__":
    X, y, classes = load_data()
    if len(X) == 0: exit("No features found. Run extract_features.py first.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Random Forest is vastly superior to NNs for this 16-feature geometry problem.
    # It does NOT overfit on augmentations and generalizes much better on unseen angles.
    rf_model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=8, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print(f"Random Forest model saved to {MODEL_SAVE_PATH}")
    
    # Quick validation print
    train_acc = rf_model.score(X_train, y_train)
    test_acc = rf_model.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")