
"""Evaluation metrics and plots."""
import numpy as np
import json, os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

class Evaluator:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = np.array(y_true)
        self.y_pred_proba = np.array(y_pred_proba)
        self.y_pred = (self.y_pred_proba >= 0.5).astype(int)

    def compute_metrics(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / cm.sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc
        }, cm, (fpr, tpr)

    def save_reports(self, out_dir='reports'):
        os.makedirs(out_dir, exist_ok=True)
        metrics, cm, roc_data = self.compute_metrics()

        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Confusion matrix
        plt.figure()
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xticks([0,1], ['No Cavitation', 'Cavitation'])
        plt.yticks([0,1], ['No Cavitation', 'Cavitation'])
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center')
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=200)
        plt.close()

        # ROC curve
        fpr, tpr = roc_data
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.2f}')
        plt.plot([0,1], [0,1], '--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=200)
        plt.close()
