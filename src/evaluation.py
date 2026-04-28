# ============================================================
# src/evaluation.py
# ============================================================
# Responsabilité : évaluer les modèles ML entraînés et
# produire des visualisations claires des performances.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Modèle") -> dict:
    """
    Calcule et affiche les métriques de performance d'un modèle.

    Métriques choisies pour la classification binaire déséquilibrée :
      - Accuracy  : taux global de bonnes prédictions (à interpréter avec prudence)
      - F1-Score  : moyenne harmonique précision/rappel (adaptée au déséquilibre)
      - AUC-ROC   : capacité à discriminer les deux classes

    Args:
        model: Modèle sklearn entraîné (doit avoir predict() et predict_proba()).
        X_test: Features de test.
        y_test: Vraies étiquettes de test.
        model_name: Nom du modèle pour l'affichage.

    Returns:
        dict: {'accuracy': ..., 'f1': ..., 'auc': ...}
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilité de la classe positive (1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"  📊 {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Rapport de classification détaillé :")
    print(classification_report(y_test, y_pred,
                                 target_names=['Low Risk (0)', 'High Risk (1)']))

    return {'model_name': model_name, 'accuracy': acc, 'f1': f1, 'auc': auc}


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Modèle"):
    """
    Affiche la matrice de confusion du modèle.

        La matrice de confusion montre :
            - Vrais Positifs (TP) : commandes à haut risque correctement prédites
            - Vrais Négatifs (TN) : commandes à faible risque correctement prédites
            - Faux Positifs (FP) : commandes à faible risque prédites comme haut risque
            - Faux Négatifs (FN) : commandes à haut risque prédites comme faible risque

    Args:
        model: Modèle sklearn entraîné.
        X_test, y_test: Données de test.
        model_name: Nom du modèle pour le titre.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Low Risk (0)', 'High Risk (1)'],
        yticklabels=['Low Risk (0)', 'High Risk (1)']
    )
    plt.title(f'Matrice de Confusion — {model_name}', fontsize=13, fontweight='bold')
    plt.ylabel('Vrai label')
    plt.xlabel('Label prédit')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_dict: dict, X_test, y_test):
    """
    Trace les courbes ROC de plusieurs modèles sur le même graphique.
    Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle.
    La diagonale représente un classifieur aléatoire (AUC = 0.5).

    Args:
        models_dict (dict): {'nom_modèle': model_sklearn, ...}
        X_test, y_test: Données de test.
    """
    plt.figure(figsize=(8, 6))

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

    # Diagonale = classifieur aléatoire (référence)
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.500)', linewidth=1)

    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR / Recall)')
    plt.title('Comparaison des courbes ROC', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list, model_name: str = "Random Forest",
                             top_n: int = 15):
    """
    Affiche les features les plus importantes selon le modèle.
    Disponible pour les modèles à base d'arbres (RandomForest, XGBoost, etc.)

    Args:
        model: Modèle sklearn avec attribut feature_importances_.
        feature_names: Liste des noms des features (dans le même ordre que X).
        model_name: Nom du modèle pour le titre.
        top_n: Nombre de features à afficher (défaut: 15).
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️  Le modèle '{model_name}' ne supporte pas feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(9, 6))
    sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        palette='viridis'
    )
    plt.title(f'Top {top_n} Features Importantes — {model_name}', fontsize=13, fontweight='bold')
    plt.xlabel("Importance (Gini)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def compare_models(results: list) -> pd.DataFrame:
    """
    Crée un tableau comparatif de plusieurs modèles et trace
    un graphique de comparaison des métriques.

    Args:
        results (list): Liste de dicts retournés par evaluate_model().
                        Ex: [{'model_name': 'RF', 'accuracy': 0.85, ...}, ...]

    Returns:
        pd.DataFrame: Tableau récapitulatif trié par F1-Score décroissant.
    """
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('f1', ascending=False)

    print("\n📊 Tableau Comparatif des Modèles :")
    print(df_results.to_string(index=False))

    # Graphique de comparaison
    metrics = ['accuracy', 'f1', 'auc']
    df_melted = df_results.melt(id_vars='model_name', value_vars=metrics,
                                 var_name='Métrique', value_name='Score')

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_melted, x='model_name', y='Score', hue='Métrique', palette='Set2')
    plt.title('Comparaison des modèles — Accuracy / F1-Score / AUC-ROC',
              fontsize=13, fontweight='bold')
    plt.xlabel('Modèle')
    plt.ylabel('Score')
    plt.ylim(0.5, 1.0)  # Zoom sur la zone utile
    plt.legend(title='Métrique')
    plt.tight_layout()
    plt.show()

    return df_results
