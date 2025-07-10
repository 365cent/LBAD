import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

def evaluate_predictions(log_type="all_combined"):
    """
    Loads model predictions and ground truth labels to compute classification metrics.

    This script assumes it is run from the root of the project directory (LBAD).
    """
    print(f"Evaluating predictions for log_type: {log_type}")

    # --- Path Definitions ---
    # Assumes the script is run from the project root directory.
    project_root = Path.cwd()
    predictions_path = project_root / "results" / log_type / f"results_{log_type}.pkl"
    
    # The ground truth labels are assumed to be in a corresponding file in the embeddings directory
    labels_path = project_root / "embeddings" / f"label_{log_type}.pkl"

    # --- File Existence Checks ---
    if not predictions_path.exists():
        print(f"Error: Predictions file not found at ''{predictions_path}''")
        print("Please ensure you have run the main transformer script to generate results.")
        return

    if not labels_path.exists():
        print(f"Error: Ground truth labels file not found at ''{labels_path}''")
        print("This script requires a ground truth labels file to evaluate predictions.")
        return

    # --- Load Predictions ---
    try:
        with open(predictions_path, 'rb') as f:
            results_data = pickle.load(f)
        
        y_pred = results_data.get('binary_predictions')
        classes = results_data.get('classes')

        if y_pred is None:
            print("Error: 'binary_predictions' key not found in the results file.")
            print(f"Available keys: {list(results_data.keys())}")
            return
        if classes is None:
            print("Warning: 'classes' key not found in results file. Metrics will be reported with integer labels.")
            
    except Exception as e:
        print(f"Error loading or parsing predictions file ''{predictions_path}'': {e}")
        return

    # --- Load Ground Truth Labels ---
    try:
        with open(labels_path, 'rb') as f:
            label_data = pickle.load(f)
        
        # We assume the key for the ground truth labels is 'labels'.
        # This might need to be adjusted based on how the label file was created.
        y_true = label_data.get('labels') 
        
        if y_true is None:
            print(f"Error: 'labels' key not found in the label file ''{labels_path}''.")
            print("Please ensure the label file contains the ground truth labels under the 'labels' key.")
            print(f"Available keys: {list(label_data.keys())}")
            return
            
        if classes is None:
            classes = label_data.get('classes')

    except Exception as e:
        print(f"Error loading or parsing labels file ''{labels_path}'': {e}")
        return

    # --- Data Validation and Alignment ---
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        print("Error: y_true and y_pred must be numpy arrays.")
        return
        
    if y_true.shape != y_pred.shape:
        print(f"Shape mismatch between true labels {y_true.shape} and predicted labels {y_pred.shape}.")
        # The training script might subsample data, but it doesn't save the indices.
        # If subsampling occurred, y_true would need to be filtered by the same indices.
        # Based on your logs (step_data_subsampling_all_combined.json), no subsampling was performed.
        return

    # --- Calculate and Print Metrics ---
    print("\n" + "="*50)
    print("Classification Evaluation Summary")
    print("="*50)

    # Generate classification report
    # The `target_names` argument requires a list of strings.
    target_names = classes if classes and len(classes) == y_true.shape[1] else None
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

    # Calculate overall accuracy (exact match ratio)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy (Exact Match Ratio): {accuracy:.4f}")
    
    # Calculate F1 scores for multi-label classification
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    print("\nF1 Scores:")
    print(f"  - Micro-average F1 score (global precision/recall): {f1_micro:.4f}")
    print(f"  - Macro-average F1 score (unweighted mean per label): {f1_macro:.4f}")
    print(f"  - Weighted-average F1 score (weighted mean per label): {f1_weighted:.4f}")
    print(f"  - Sample-average F1 score (per-instance metric): {f1_samples:.4f}")
    print("="*50)


if __name__ == "__main__":
    # This script is designed to be run from the root of the project.
    # It evaluates the 'all_combined' results by default.
    evaluate_predictions(log_type="all_combined")
