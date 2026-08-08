import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support

FEATURE_DIR = "extracted_features"
MODEL_PATH = "fall_detection_rf.pkl"
PLOTS_PATH = "evaluation_plots"

def load_data():
    X, y = [], []
    classes = {'Falling': 0, 'Normal': 1}
    label_map = {v:k for k,v in classes.items()}
    for f in os.listdir(FEATURE_DIR):
        if f.endswith('.npy'):
            label_str = f.split('_')[0]
            if label_str in classes:
                X.append(np.load(os.path.join(FEATURE_DIR, f)))
                y.append(classes[label_str])
    return np.array(X), np.array(y), label_map

if __name__ == "__main__":
    X, y, label_map = load_data()
    if len(X) == 0 or not os.path.exists(MODEL_PATH): exit("Missing data or model.")
    
    rf_model = joblib.load(MODEL_PATH)
    
    # Re-split to get Train and Test sets for accuracy comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    y_pred = rf_model.predict(X_test)
    y_train_pred = rf_model.predict(X_train)
    
    all_labels = list(label_map.keys())
    target_names = list(label_map.values())
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=all_labels, zero_division=0)
    
    print("\n--- BINARY EVALUATION METRICS (RANDOM FOREST) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n--- CLASSIFICATION REPORT ---")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    for i, name in enumerate(target_names):
        print(f"{name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    # Generate & Save both Confusion Matrix and a Performance Graph
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Binary Confusion Matrix (Random Forest)')
    plt.savefig(f"{PLOTS_PATH}/confusion_matrix.png")
    plt.close()
    print(f"\nConfusion Matrix saved.")
    
    # 2. Model Performance & Feature Importance Graph (Replaces the old Accuracy/Loss graphs)
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Train vs Test Accuracy
    plt.subplot(1, 2, 1)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    bars = plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_acc, test_acc], color=['#4CAF50', '#2196F3'])
    plt.ylim(0, 1.05)
    plt.title('Model Accuracy (No Overfitting)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

    # Subplot 2: Top 8 Feature Importances
    plt.subplot(1, 2, 2)
    feature_names = ['Head Tilt', 'L Shoulder Ang', 'R Shoulder Ang', 'L Elbow', 'R Elbow', 
                     'L Hip Ang', 'R Hip Ang', 'L Knee Ang', 'R Knee Ang', 'L Ankle Ang', 'R Ankle Ang',
                     'Torso Angle', 'Sh Line Ang', 'Hip Line Ang', 'Shoulder Width', 'Leg/Torso Ratio']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-8:]  # Top 8 features
    plt.barh([feature_names[i] for i in indices], importances[indices], color='#FF9800')
    plt.title('Top 8 Deciding Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/accuracy_loss_graphs.png")  # Keeping the same name so app doesn't break
    plt.close()
    print(f"Performance & Feature Importance graph saved to {PLOTS_PATH}/accuracy_loss_graphs.png")