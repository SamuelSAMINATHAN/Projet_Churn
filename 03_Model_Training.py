import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, confusion_matrix
import joblib
import importlib

# Chargement du module optimisé (Expert)
preprocessing = importlib.import_module("02_Preprocessing")
ChurnExpertPipeline = preprocessing.ChurnExpertPipeline

def train_expert_model():
    # 1. Préparation des données (Utilisation du Pipeline Expert avec Feature Engineering)
    manager = ChurnExpertPipeline('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_full = manager.load_and_prepare()
    
    X = df_full.drop('Churn', axis=1)
    y = (df_full['Churn'] == 'Yes').astype(int)
    
    # Split stratifié
    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Transformation en DataFrame Pandas (Garde les noms de colonnes pour SHAP)
    X_train, X_test = manager.process_to_df(X_train_raw, X_test_raw)
    
    # 2. Calcul du poids pour gérer le déséquilibre
    # scale_pos_weight = total_negative / total_positive
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 3. Configuration XGBoost avec Early Stopping
    # On définit un set de validation pour monitorer l'entraînement
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=1000, # On met beaucoup, l'early stopping coupera
        random_state=42,
        scale_pos_weight=ratio,
        eval_metric='aucpr', # Precision-Recall AUC est meilleur pour le churn déséquilibré
        early_stopping_rounds=50 
    )

    # 4. Optimisation par recherche aléatoire (RandomizedSearch)
    param_dist = {
        'max_depth': [3, 4, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 1], # Régularisation L1
        'reg_lambda': [1, 5, 10]  # Régularisation L2
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("🚀 Optimisation du Stratège en cours...")
    search = RandomizedSearchCV(
        clf, param_distributions=param_dist, 
        n_iter=15, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    
    # On passe un eval_set pour que l'early stopping fonctionne au sein du search (XGBoost le permet)
    search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    best_model = search.best_estimator_
    
    # 5. Évaluation Multi-niveaux
    y_probs = best_model.predict_proba(X_test)[:, 1]
    
    # Astuce Expert : Ajuster le seuil (Threshold)
    # Par défaut c'est 0.5. On le descend à 0.4 pour capturer PLUS de churners (Recall)
    threshold = 0.4
    y_pred_custom = (y_probs >= threshold).astype(int)

    print("\n" + "="*30)
    print("📈 PERFORMANCE DU MODÈLE")
    print("="*30)
    print(f"Meilleurs Paramètres : {search.best_params_}")
    print(f"ROC-AUC : {roc_auc_score(y_test, y_probs):.4f}")
    print(f"Recall (Capture du Churn) : {recall_score(y_test, y_pred_custom):.4f}")
    print(f"F1-Score (Seuil {threshold}) : {f1_score(y_test, y_pred_custom):.4f}")
    
    print("\n--- Confusion Matrix (Seuil 0.4) ---")
    print(confusion_matrix(y_test, y_pred_custom))

    # 6. Sauvegarde Industrielle
    # Joblib est plus efficace que pickle pour les gros modèles
    joblib.dump(best_model, 'outputs/model_xgboost_expert.joblib')
    joblib.dump(manager.preprocessor, 'outputs/preprocessor.joblib')
    # On sauvegarde aussi les noms de colonnes pour SHAP
    joblib.dump(X_train.columns.tolist(), 'outputs/feature_names.joblib')

    print("\n✅ Modèle et Preprocessor sauvegardés dans /outputs/")

if __name__ == "__main__":
    train_expert_model()