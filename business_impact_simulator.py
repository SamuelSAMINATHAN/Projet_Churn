import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import importlib

# 1. Chargement des données réelles du projet
def load_real_data():
    # Chargement du modèle et des données de test
    model = joblib.load('outputs/model_xgboost_expert.joblib')
    feature_names = joblib.load('outputs/feature_names.joblib')
    
    # Utilisation du pipeline pour reconstruire le set de test
    preprocessing = importlib.import_module("02_Preprocessing")
    manager = preprocessing.ChurnExpertPipeline('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_full = manager.load_and_prepare()
    
    from sklearn.model_selection import train_test_split
    X = df_full.drop('Churn', axis=1)
    y = (df_full['Churn'] == 'Yes').astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    _, X_test_df = manager.process_to_df(X_train_raw, X_test_raw)
    
    # On récupère les probabilités réelles du modèle XGBoost
    y_probs = model.predict_proba(X_test_df)[:, 1]
    return y_test.values, y_probs

def simulate_roi_advanced(y_true, y_probs, threshold, avg_clv=2000, retention_cost=150, success_rate=0.4):
    """
    Simulation améliorée avec paramètres business ajustables.
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # CALCULS FINANCIERS
    # Coût de base (si on ne fait rien)
    cost_no_action = np.sum(y_true) * avg_clv
    
    # Coûts avec le modèle
    marketing_spend = (tp + fp) * retention_cost # On paie pour les TP et les FP (fausses alertes)
    saved_revenue = tp * success_rate * avg_clv # Seuls les TP qui acceptent l'offre rapportent
    lost_revenue = (fn * avg_clv) + (tp * (1 - success_rate) * avg_clv) # FN + Echecs de rétention
    
    total_cost_model = marketing_spend + lost_revenue
    net_profit_saved = cost_no_action - total_cost_model
    
    return {
        'threshold': threshold,
        'net_profit_saved': net_profit_saved,
        'marketing_spend': marketing_spend,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / np.sum(y_true),
        'tp': tp, 'fp': fp, 'fn': fn
    }

def run_business_optimization(y_true, y_probs):
    # Paramètres d'entrée (ajustables selon les réunions marketing)
    CLV = 1500 # Valeur moyenne d'un client sur sa durée de vie
    COST = 100  # Coût d'un mois offert ou d'une promo
    EFFICIENCY = 0.3 # 30% des clients ciblés décident de rester
    
    thresholds = np.linspace(0.05, 0.95, 50)
    results = [simulate_roi_advanced(y_true, y_probs, t, CLV, COST, EFFICIENCY) for t in thresholds]
    df = pd.DataFrame(results)
    
    # Trouver le meilleur seuil
    best_idx = df['net_profit_saved'].idxmax()
    best_t = df.loc[best_idx, 'threshold']
    max_gain = df.loc[best_idx, 'net_profit_saved']
    
    # VISUALISATION PROFESSIONNELLE
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Courbe de Profit
    color = 'tab:green'
    ax1.set_xlabel('Seuil de probabilité (Modèle)')
    ax1.set_ylabel('Profit Net Économisé (€)', color=color)
    ax1.plot(df['threshold'], df['net_profit_saved'], color=color, linewidth=3, label='Profit Sauvé')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.fill_between(df['threshold'], df['net_profit_saved'], alpha=0.2, color=color)
    
    # Courbe des coûts marketing (Faux Positifs)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Dépenses Marketing Inutiles (€)', color=color)
    ax2.plot(df['threshold'], df['marketing_spend'], color=color, linestyle='--', label='Budget Engagement')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Optimisation du Seuil Business\nMax Gain: {max_gain:,.0f}€ au seuil {best_t:.2f}', fontsize=15)
    ax1.axvline(best_t, color='black', linestyle=':', label=f'Optimal (t={best_t:.2f})')
    fig.tight_layout()
    plt.show()
    
    print(f"💰 STRATÉGIE OPTIMALE :")
    print(f"En réglant le modèle sur un seuil de {best_t:.2f} :")
    print(f"- On identifie {df.loc[best_idx, 'tp']} vrais départs potentiels.")
    print(f"- On accepte {df.loc[best_idx, 'fp']} fausses alertes (coût maîtrisé).")
    print(f"- GAIN NET POUR L'ENTREPRISE : {max_gain:,.2f} €")

if __name__ == "__main__":
    try:
        y_true, y_probs = load_real_data()
        run_business_optimization(y_true, y_probs)
    except FileNotFoundError:
        print("Erreur : Assurez-vous d'avoir lancé les scripts 02 et 03 avant celui-ci.")