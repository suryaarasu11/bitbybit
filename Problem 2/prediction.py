import os
import pandas as pd
import numpy as np
import pickle

def load_trained_model(model_path='cnn_classifier_model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def extract_features_from_chunk(chunk, col_index=0):
    """Extract basic statistics from a chunk of counts."""
    col = pd.to_numeric(chunk.iloc[:, col_index], errors='coerce').dropna().values
    if len(col) == 0:
        return None
    return {
        'mean': np.mean(col),
        'std': np.std(col),
        'min': np.min(col),
        'max': np.max(col),
        'median': np.median(col)
    }

def features_from_csv(csv_path, chunk_size=100, col_index=0):
    """Split CSV into chunks and extract features from each chunk."""
    df = pd.read_csv(csv_path, header=None)
    features_list = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        if len(chunk) < chunk_size:
            continue
        feats = extract_features_from_chunk(chunk, col_index)
        if feats is not None:
            features_list.append(feats)
    if not features_list:
        return None
    return pd.DataFrame(features_list)[['mean', 'std', 'min', 'max', 'median']]

def predict_csv(csv_path, model, chunk_size=100):
    """Predict the model for a single CSV file."""
    features_df = features_from_csv(csv_path, chunk_size)
    if features_df is None:
        return None, None, 0
    X = features_df[['mean', 'std', 'min', 'max', 'median']]
    predictions = model.predict(X)
    unique, counts = np.unique(predictions, return_counts=True)
    most_common_model = unique[np.argmax(counts)]
    confidence = np.max(counts) / len(predictions)
    return most_common_model, confidence, len(predictions)

def predict_aggregated(run_csv_files, model, chunk_size=100):
    """Aggregate predictions across all CSVs of a run using soft-voting."""
    all_features = []
    for csv_file in run_csv_files:
        df = features_from_csv(csv_file, chunk_size)
        if df is not None:
            all_features.append(df)
    if not all_features:
        return None, None
    X_all = pd.concat(all_features, ignore_index=True)
    probs = model.predict_proba(X_all)
    avg_probs = probs.mean(axis=0)
    final_class = model.classes_[np.argmax(avg_probs)]
    confidence = np.max(avg_probs)
    return final_class, confidence

if __name__ == "__main__":
    model = load_trained_model()
    traces_folder = 'new_traces_clean'

    # Group CSVs by run prefix (e.g., modelX_branch-misses → modelX)
    runs = {}
    for file in os.listdir(traces_folder):
        if file.endswith('.csv'):
            run_name = file.split('_')[0]
            runs.setdefault(run_name, []).append(os.path.join(traces_folder, file))

    for run_name, files in runs.items():
        print(f"\n=== RUN: {run_name} ===")
        # 1️⃣ Per-CSV predictions
        for file in sorted(files):
            pred_model, conf, chunks = predict_csv(file, model)
            if pred_model:
                print(f"{os.path.basename(file)}: Predicted = {pred_model}, Confidence = {conf:.2%}, Chunks = {chunks}")
            else:
                print(f"{os.path.basename(file)}: Prediction failed")

        # 2️⃣ Aggregated run prediction
        agg_model, agg_conf = predict_aggregated(files, model)
        if agg_model:
            print(f"Aggregated Run Prediction: {agg_model}, Confidence = {agg_conf:.2%}")
        else:
            print("Aggregated Run Prediction: Failed")
