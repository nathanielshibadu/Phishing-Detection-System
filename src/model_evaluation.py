import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import joblib

# Define project dirs relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set(style="whitegrid")


def load_features_labels(data_dir=DATA_DIR):
    """Load features.csv and labels.csv from data_dir and return (X, y)."""
    X = pd.read_csv(os.path.join(data_dir, 'features.csv'))
    y = pd.read_csv(os.path.join(data_dir, 'labels.csv')).squeeze()
    return X, y


def load_model(model_path):
    """Load model from path. Supports joblib/pkl, xgboost json, keras h5."""
    ext = os.path.splitext(model_path)[1].lower()
    try:
        if ext in ('.pkl', '.joblib'):
            return joblib.load(model_path)
        elif ext == '.json' or ext == '.model':
            # try xgboost
            try:
                from xgboost import XGBClassifier
                m = XGBClassifier()
                m.load_model(model_path)
                return m
            except Exception:
                raise
        elif ext in ('.h5', '.keras'):
            try:
                from tensorflow.keras.models import load_model
                return load_model(model_path)
            except Exception:
                raise
        else:
            # fallback to joblib attempt
            return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_path}: {e}")


def _get_probs(model, X):
    """Return probability estimates or decision function as scores for positive class."""
    # prefer predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # if binary, return positive class probability
        if probs.shape[1] == 2:
            return probs[:, 1]
        # for multiclass, return full matrix
        return probs
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # if binary decision_function returns shape (n,), convert via sigmoid-like scaling for ROC
        if scores.ndim == 1:
            return scores
        # multiclass
        return scores
    # fallback: use predictions as 0/1
    preds = model.predict(X)
    return np.asarray(preds)


def evaluate_model(model, X, y, model_name="model", output_dir=REPORTS_DIR):
    """Compute metrics, generate plots and return metrics dict."""
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y).ravel()

    t0 = time.time()
    y_pred = model.predict(X_arr)
    inference_time = time.time() - t0

    # Try to get scores for ROC/PR
    try:
        y_score = _get_probs(model, X_arr)
    except Exception:
        y_score = None

    acc = accuracy_score(y_arr, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_arr, y_pred, average='binary', zero_division=0)
    report = classification_report(y_arr, y_pred, zero_division=0, output_dict=True)

    metrics = {
        'model': model_name,
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'inference_time_sec': float(inference_time)
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_arr, y_pred)
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    # ROC curve (binary)
    if y_score is not None:
        try:
            # if multiclass probs matrix, try to compute micro-average ROC if possible
            if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 2):
                if y_score.ndim == 2:
                    y_score_pos = y_score[:, 1]
                else:
                    y_score_pos = y_score
                fpr, tpr, _ = roc_curve(y_arr, y_score_pos)
                roc_auc = roc_auc_score(y_arr, y_score_pos)
                fig_roc, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], '--', color='gray')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve - {model_name}')
                ax.legend(loc='lower right')
                roc_path = os.path.join(output_dir, f'{model_name}_roc.png')
                fig_roc.tight_layout()
                fig_roc.savefig(roc_path)
                plt.close(fig_roc)
                metrics['roc_auc'] = float(roc_auc)
        except Exception:
            pass

        # Precision-Recall curve (binary)
        try:
            if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 2):
                y_score_pos = y_score[:, 1] if y_score.ndim == 2 else y_score
                prec, rec, _ = precision_recall_curve(y_arr, y_score_pos)
                auc_pr = np.trapz(prec, rec)
                fig_pr, ax = plt.subplots(figsize=(5, 4))
                ax.plot(rec, prec, label=f'PR AUC = {auc_pr:.3f}')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision-Recall Curve - {model_name}')
                ax.legend(loc='lower left')
                pr_path = os.path.join(output_dir, f'{model_name}_precision_recall.png')
                fig_pr.tight_layout()
                fig_pr.savefig(pr_path)
                plt.close(fig_pr)
                metrics['pr_auc'] = float(auc_pr)
        except Exception:
            pass

    # Feature importance for tree-based models
    try:
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            feature_names = X.columns.tolist()
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
            fi_df = fi_df.sort_values('importance', ascending=False).head(30)
            fig_fi, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(fi_df))))
            sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
            ax.set_title(f'Feature Importances - {model_name}')
            fi_path = os.path.join(output_dir, f'{model_name}_feature_importance.png')
            fig_fi.tight_layout()
            fig_fi.savefig(fi_path)
            plt.close(fig_fi)
            metrics['has_feature_importance'] = True
    except Exception:
        metrics['has_feature_importance'] = False

    # Save classification report as csv
    try:
        cr_df = pd.DataFrame(report).transpose()
        cr_path = os.path.join(output_dir, f'{model_name}_classification_report.csv')
        cr_df.to_csv(cr_path, index=True)
    except Exception:
        pass

    return metrics


def evaluate_model_from_path(model_path, X, y, output_dir=REPORTS_DIR):
    """Load a model from path and evaluate it. Returns metrics dict."""
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Skipping {model_path}: {e}")
        return None

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    metrics = evaluate_model(model, X, y, model_name=model_name, output_dir=output_dir)
    print(f"Evaluated {model_name}: accuracy={metrics.get('accuracy'):.4f}")
    return metrics


def main(model_paths=None):
    X, y = load_features_labels()
    metrics_list = []

    if model_paths is None:
        # evaluate all models in MODELS_DIR with common extensions
        candidates = []
        for f in os.listdir(MODELS_DIR):
            if f.lower().endswith(('.pkl', '.joblib', '.json', '.h5', '.keras', '.model')):
                candidates.append(os.path.join(MODELS_DIR, f))
        model_paths = sorted(candidates)

    for mp in model_paths:
        m = evaluate_model_from_path(mp, X, y)
        if m:
            metrics_list.append(m)

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(os.path.join(REPORTS_DIR, 'models_evaluation_summary.csv'), index=False)
        print(f"Saved evaluation summary to {os.path.join(REPORTS_DIR, 'models_evaluation_summary.csv')}")
    else:
        print("No models evaluated.")


if __name__ == "__main__":
    main()

# ...existing code...

def evaluate_model(model, X_test, y_test):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score
    )
    from sklearn.preprocessing import label_binarize

    # Ensure arrays
    X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_true = y_test.values.ravel() if hasattr(y_test, "values") else np.asarray(y_test).ravel()

    # Predictions
    y_pred = model.predict(X_arr)

    # Basic metrics (use weighted average to handle class imbalance / multiclass)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    # ROC AUC (handle binary and multiclass)
    roc_auc = None
    try:
        # obtain score/probabilities if available
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_arr)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_arr)
            # decision_function might return shape (n_samples,) for binary; make it (n_samples,1)
            if y_score.ndim == 1:
                y_score = y_score
        else:
            y_score = None

        if y_score is not None:
            # binary case: single-dim scores or two-column probabilities
            if getattr(y_score, "ndim", 1) == 1 or (hasattr(y_score, "ndim") and y_score.ndim == 2 and y_score.shape[1] == 2):
                if hasattr(y_score, "ndim") and y_score.ndim == 2:
                    y_score_pos = y_score[:, 1]
                else:
                    y_score_pos = y_score
                roc_auc = roc_auc_score(y_true, y_score_pos)
            else:
                # multiclass: binarize true labels and compute macro OVR AUC
                classes = np.unique(y_true)
                y_true_b = label_binarize(y_true, classes=classes)
                # If score columns match classes, compute multiclass AUC
                if hasattr(y_score, "ndim") and y_score.ndim == 2 and y_score.shape[1] == y_true_b.shape[1]:
                    roc_auc = roc_auc_score(y_true_b, y_score, average='macro', multi_class='ovr')
    except Exception:
        roc_auc = None

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc) if roc_auc is not None else None
    }

    # Nicely formatted printout
    print("Evaluation results:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC AUC  : {metrics['roc_auc']:.4f}")
    else:
        print("  ROC AUC  : N/A (model does not provide probability/score or computation failed)")

    return metrics

# ...existing code...

# ...existing code...

def measure_inference_time(model, sample, iterations=100, warmup=5):
    """
    Measure average inference latency for a single sample or small batch.
    Returns mean latency in milliseconds.
    """
    import numpy as np

    # Prepare input array
    if hasattr(sample, "values"):
        X = sample.values
    else:
        X = np.asarray(sample)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Warm-up runs
    for _ in range(max(1, warmup)):
        try:
            model.predict(X)
        except Exception:
            # some models (e.g., tensorflow) may require different call; try calling directly
            try:
                model(X)
            except Exception:
                pass

    # Timed runs
    times = []
    for _ in range(max(1, iterations)):
        t0 = time.time()
        try:
            model.predict(X)
        except Exception:
            try:
                model(X)
            except Exception:
                # if predict fails, skip timing this iteration
                continue
        times.append(time.time() - t0)

    if len(times) == 0:
        return None

    mean_latency_ms = float(np.mean(times) * 1000.0)
    return mean_latency_ms

# ...existing code...
# ...existing code...

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Plot and save a confusion matrix for y_true vs y_pred.
    Saved to: <project_root>/results/confusion_matrices/{model_name}_confusion_matrix.png
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    # Ensure arrays
    y_true_arr = y_true.values.ravel() if hasattr(y_true, "values") else np.asarray(y_true).ravel()
    y_pred_arr = y_pred.values.ravel() if hasattr(y_pred, "values") else np.asarray(y_pred).ravel()

    # Determine class labels (sorted)
    labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))

    # Compute confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    # Prepare output directory
    out_dir = os.path.join(BASE_DIR, 'results', 'confusion_matrices')
    os.makedirs(out_dir, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    fig.tight_layout()

    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in model_name).strip()
    file_path = os.path.join(out_dir, f'{safe_name}_confusion_matrix.png')
    fig.savefig(file_path)
    plt.close(fig)

    return file_path

# ...existing code...
# ...existing code...

def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """
    Plot and save ROC curve(s).
    Saves to: <project_root>/results/roc_curves/{model_name}_roc.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Ensure arrays
    y_true_arr = y_true.values.ravel() if hasattr(y_true, "values") else np.asarray(y_true).ravel()
    y_score = y_proba
    if hasattr(y_score, "values"):
        y_score = y_score.values
    y_score = np.asarray(y_score)

    out_dir = os.path.join(BASE_DIR, 'results', 'roc_curves')
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    try:
        # Binary case (1D scores or 2-column probabilities)
        if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 2):
            if y_score.ndim == 2:
                y_score_pos = y_score[:, 1]
            else:
                y_score_pos = y_score
            fpr, tpr, _ = roc_curve(y_true_arr, y_score_pos)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        else:
            # Multiclass: compute per-class ROC and plot; compute macro AUC
            classes = np.unique(y_true_arr)
            y_true_b = label_binarize(y_true_arr, classes=classes)
            aucs = []
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_true_b[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                ax.plot(fpr, tpr, label=f'class {cls} (AUC = {roc_auc:.3f})')
            macro_auc = float(np.mean(aucs)) if len(aucs) > 0 else 0.0
            ax.set_title(f'ROC Curve - {model_name} (macro AUC = {macro_auc:.3f})')

        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.set_title(ax.get_title() or f'ROC Curve - {model_name}')

        safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in model_name).strip()
        file_path = os.path.join(out_dir, f'{safe_name}_roc.png')
        fig.tight_layout()
        fig.savefig(file_path)
        plt.close(fig)
        return file_path
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Failed to plot ROC curve for {model_name}: {e}")

# ...existing code...
# filepath: c:\Users\Noel\Phishing-Detection-System\src\model_evaluation.py
# ...existing code...

def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """
    Plot and save ROC curve(s).
    Saves to: <project_root>/results/roc_curves/{model_name}_roc.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Ensure arrays
    y_true_arr = y_true.values.ravel() if hasattr(y_true, "values") else np.asarray(y_true).ravel()
    y_score = y_proba
    if hasattr(y_score, "values"):
        y_score = y_score.values
    y_score = np.asarray(y_score)

    out_dir = os.path.join(BASE_DIR, 'results', 'roc_curves')
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    try:
        # Binary case (1D scores or 2-column probabilities)
        if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 2):
            if y_score.ndim == 2:
                y_score_pos = y_score[:, 1]
            else:
                y_score_pos = y_score
            fpr, tpr, _ = roc_curve(y_true_arr, y_score_pos)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        else:
            # Multiclass: compute per-class ROC and plot; compute macro AUC
            classes = np.unique(y_true_arr)
            y_true_b = label_binarize(y_true_arr, classes=classes)
            aucs = []
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_true_b[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                ax.plot(fpr, tpr, label=f'class {cls} (AUC = {roc_auc:.3f})')
            macro_auc = float(np.mean(aucs)) if len(aucs) > 0 else 0.0
            ax.set_title(f'ROC Curve - {model_name} (macro AUC = {macro_auc:.3f})')

        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.set_title(ax.get_title() or f'ROC Curve - {model_name}')

        safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in model_name).strip()
        file_path = os.path.join(out_dir, f'{safe_name}_roc.png')
        fig.tight_layout()
        fig.savefig(file_path)
        plt.close(fig)
        return file_path
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Failed to plot ROC curve for {model_name}: {e}")

# ...existing code...
# ...existing code...

def main():
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Locate model file (prefer random_forest.pkl)
    preferred = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if os.path.exists(preferred):
        model_path = preferred
    else:
        # fallback: pick first supported model in MODELS_DIR
        candidates = [f for f in os.listdir(MODELS_DIR)
                      if f.lower().endswith(('.pkl', '.joblib', '.json', '.h5', '.keras', '.model'))]
        if not candidates:
            print(f"No models found in {MODELS_DIR}. Exiting.")
            return
        model_path = os.path.join(MODELS_DIR, sorted(candidates)[0])

    print(f"Loading model: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return

    # Load test data if available, otherwise split from full dataset
    test_features_path = os.path.join(DATA_DIR, 'test_features.csv')
    test_labels_path = os.path.join(DATA_DIR, 'test_labels.csv')

    if os.path.exists(test_features_path) and os.path.exists(test_labels_path):
        X_test = pd.read_csv(test_features_path)
        y_test = pd.read_csv(test_labels_path).squeeze()
        print(f"Loaded test data from {test_features_path} and {test_labels_path}")
    else:
        X_all, y_all = load_features_labels()
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            print("No separate test files found — using an 80/20 stratified split from full dataset.")
        except Exception:
            # fallback to non-stratified split if stratify fails
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )
            print("Stratified split failed, used non-stratified 80/20 split.")

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Evaluate using evaluate_model (prints metrics and returns dict)
    try:
        metrics = evaluate_model(model, X_test, y_test)
    except Exception as e:
        print(f"evaluate_model failed: {e}")
        metrics = None

    # Predictions and probabilities for additional plots
    try:
        X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
        y_pred = model.predict(X_arr)
    except Exception as e:
        print(f"Model prediction failed: {e}")
        return

    # Try to obtain probabilities / scores for ROC
    y_proba = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_arr)
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_arr)
    except Exception:
        y_proba = None

    # Measure inference time (use one sample or small batch)
    sample = X_test.iloc[0] if hasattr(X_test, "iloc") else X_test[:1]
    latency_ms = measure_inference_time(model, sample)
    if latency_ms is not None:
        print(f"Average inference latency: {latency_ms:.3f} ms")
    else:
        print("Inference timing unavailable.")

    # Plot and save confusion matrix
    try:
        cm_path = plot_confusion_matrix(y_test, pd.Series(y_pred), model_name=model_name)
        print(f"Confusion matrix saved to: {cm_path}")
    except Exception as e:
        print(f"Failed to plot confusion matrix: {e}")

    # Plot ROC if probabilities/scores are available
    try:
        if y_proba is not None:
            roc_path = plot_roc_curve(y_test, y_proba, model_name=model_name)
            print(f"ROC curve saved to: {roc_path}")
        else:
            print("Skipping ROC plot: model does not provide probability or decision scores.")
    except Exception as e:
        print(f"Failed to plot ROC curve: {e}")

    # Summarize
    if metrics is not None:
        print("Final metrics summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
# filepath: c:\Users\Noel\Phishing-Detection-System\src\model_evaluation.py
# ...existing code...

def main():
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Locate model file (prefer random_forest.pkl)
    preferred = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if os.path.exists(preferred):
        model_path = preferred
    else:
        # fallback: pick first supported model in MODELS_DIR
        candidates = [f for f in os.listdir(MODELS_DIR)
                      if f.lower().endswith(('.pkl', '.joblib', '.json', '.h5', '.keras', '.model'))]
        if not candidates:
            print(f"No models found in {MODELS_DIR}. Exiting.")
            return
        model_path = os.path.join(MODELS_DIR, sorted(candidates)[0])

    print(f"Loading model: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return

    # Load test data if available, otherwise split from full dataset
    test_features_path = os.path.join(DATA_DIR, 'test_features.csv')
    test_labels_path = os.path.join(DATA_DIR, 'test_labels.csv')

    if os.path.exists(test_features_path) and os.path.exists(test_labels_path):
        X_test = pd.read_csv(test_features_path)
        y_test = pd.read_csv(test_labels_path).squeeze()
        print(f"Loaded test data from {test_features_path} and {test_labels_path}")
    else:
        X_all, y_all = load_features_labels()
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            print("No separate test files found — using an 80/20 stratified split from full dataset.")
        except Exception:
            # fallback to non-stratified split if stratify fails
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )
            print("Stratified split failed, used non-stratified 80/20 split.")

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Evaluate using evaluate_model (prints metrics and returns dict)
    try:
        metrics = evaluate_model(model, X_test, y_test)
    except Exception as e:
        print(f"evaluate_model failed: {e}")
        metrics = None

    # Predictions and probabilities for additional plots
    try:
        X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
        y_pred = model.predict(X_arr)
    except Exception as e:
        print(f"Model prediction failed: {e}")
        return

    # Try to obtain probabilities / scores for ROC
    y_proba = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_arr)
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_arr)
    except Exception:
        y_proba = None

    # Measure inference time (use one sample or small batch)
    sample = X_test.iloc[0] if hasattr(X_test, "iloc") else X_test[:1]
    latency_ms = measure_inference_time(model, sample)
    if latency_ms is not None:
        print(f"Average inference latency: {latency_ms:.3f} ms")
    else:
        print("Inference timing unavailable.")

    # Plot and save confusion matrix
    try:
        cm_path = plot_confusion_matrix(y_test, pd.Series(y_pred), model_name=model_name)
        print(f"Confusion matrix saved to: {cm_path}")
    except Exception as e:
        print(f"Failed to plot confusion matrix: {e}")

    # Plot ROC if probabilities/scores are available
    try:
        if y_proba is not None:
            roc_path = plot_roc_curve(y_test, y_proba, model_name=model_name)
            print(f"ROC curve saved to: {roc_path}")
        else:
            print("Skipping ROC plot: model does not provide probability or decision scores.")
    except Exception as e:
        print(f"Failed to plot ROC curve: {e}")

    # Summarize
    if metrics is not None:
        print("Final metrics summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()