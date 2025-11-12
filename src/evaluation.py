"""
Evaluación: Accuracy, F1, ROC, SHAP, métricas de negocio.
"""
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import shap
import numpy as np

class ModelEvaluator:
    
    def evaluate_ensemble(self, y_test, y_pred, y_proba):
        """Métri cas clásicas."""
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"Accuracy: {acc:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"Precision: {tp / (tp + fp):.3f}")
        print(f"Recall: {tp / (tp + fn):.3f}")
        print(f"False Positive Rate: {fp / (fp + tn):.3f}")
        
        return {'accuracy': acc, 'f1': f1}
    
    def plot_roc_curve(self, y_test, y_proba):
        """Dibuja curva ROC."""
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('results/roc_curve.png', dpi=150)
        plt.close()
    
    def explain_with_shap(self, model, X_sample):
        """Explica qué contribuyó a la predicción (SHAP)."""
        explainer = shap.KernelExplainer(model.predict, X_sample[:100])
        shap_values = explainer.shap_values(X_sample)
        
        # Visualiza importancia de features
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.savefig('results/shap_feature_importance.png', dpi=150)
        plt.close()
        
        print("✓ SHAP explicabilidad guardada en results/")

# USO:
# evaluator = ModelEvaluator()
# evaluator.evaluate_ensemble(y_test, y_pred, y_proba)
# evaluator.plot_roc_curve(y_test, y_proba)
# evaluator.explain_with_shap(ensemble, X_test)
