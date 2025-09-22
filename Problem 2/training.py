import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_folder = 'profiled_data'
chunk_size = 100

files_models = {
    'alexnet_data.csv': 'AlexNet',
    'vgg_data.csv': 'VGG',
    'resnet_data.csv': 'ResNet',
    'densenet_data.csv': 'DenseNet',
    'inception_v3_data.csv': 'InceptionV3',
    'mobilenet_v2_data.csv': 'MobileNetV2',
    'shufflenet_v2_x1_0_data.csv': 'ShuffleNetV2'
}

def extract_features(chunk):
    col = chunk.iloc[:, 0].dropna().values
    return {
        'mean': np.mean(col),
        'std': np.std(col),
        'min': np.min(col),
        'max': np.max(col),
        'median': np.median(col)
    }

# --- Load data and extract features ---
all_features = []
for filename, model in files_models.items():
    path = os.path.join(data_folder, filename)
    print(f'Processing {filename} for {model}')
    df = pd.read_csv(path, header=None)
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        if len(chunk) == chunk_size:
            feats = extract_features(chunk)
            feats['model'] = model
            all_features.append(feats)

features_dataset = pd.DataFrame(all_features)

X = features_dataset[['mean', 'std', 'min', 'max', 'median']]
y = features_dataset['model']

# --- Cross-validation ---
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Accuracy per fold
scores = cross_val_score(clf, X, y, cv=cv)
print("\nCross-validation accuracy scores:", scores)
print("Mean CV Accuracy: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))

# Predictions for confusion matrix
y_pred_cv = cross_val_predict(clf, X, y, cv=cv)

print("\nClassification Report (CV):")
print(classification_report(y, y_pred_cv))

# --- Confusion Matrix ---
labels = np.unique(y)  # use actual labels
cm = confusion_matrix(y, y_pred_cv, labels=labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues")
plt.title("Confusion Matrix (5-fold CV)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# --- Save final model (trained on all data) ---
clf.fit(X, y)
with open('cnn_classifier_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("\n✓ Final model trained on full dataset and saved as 'cnn_classifier_model.pkl'")

