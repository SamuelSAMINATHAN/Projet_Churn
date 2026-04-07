# Projet Telco Customer Churn

## Objectif du projet
L'objectif de ce projet est de transformer une analyse de données historique en un outil d'anticipation stratégique. Au lieu de simplement constater le départ des clients, nous avons mis en place un système capable de prédire **qui va partir demain et pourquoi**, permettant ainsi des interventions marketing ciblées et rentables.

**Dataset :** Telco Customer Churn (Données comportementales de télécoms).  
**Stack Technique :** Python (Pandas, Scikit-Learn), XGBoost, SHAP (Interprétabilité).

---

##  1. Analyse Exploratoire des Données (EDA)
L'analyse a révélé des points de friction majeurs dans le parcours client.

### Distribution du Churn
Le dataset présente un déséquilibre avec environ **26.5% de churn**. 
![Distribution du Churn](reports/Distribution%20du%20Churn.png)

### Leviers de "Stickiness" (Adhérence)
*   **La Tenure :** Le risque de départ est maximal durant les 6 premiers mois. Les clients qui dépassent le cap des 20 mois ont une probabilité de fidélisation bien plus élevée.
![Distribution de la Tenure](reports/Distribution%20de%20la%20Tenure%20-%20Retained%20vs%20Churned.png)

*   **Le Contrat :** Le contrat "Month-to-month" est le principal vecteur de churn. Les contrats à long terme (1 ou 2 ans) agissent comme une barrière naturelle au départ.
![Impact du Contrat](reports/Impact%20du%20Contrat%20sur%20le%20Churn.png)

*   **Services et Support :** Les clients utilisant la **Fibre Optique** churnent anormalement plus que ceux en DSL, souvent par manque de support technique ou de sécurité en ligne.
![Segmentation des Services](reports/Segmentation%20des%20Services.png)
![Taux de Churn Services](reports/Taux%20de%20Churn%20en%20fonction%20du%20nombre%20de%20services%20souscrits.png)

---

##  2. Data Engineering & Preprocessing
Pour améliorer la précision du modèle, nous avons créé ces variables  :
- **TotalServices :** Somme des options souscrites (indicateur de dépendance à l'écosystème).
- **ChargePerMonth_Ratio :** Détection des anomalies de facturation historique.
- **IsFiber :** Isolation du segment à haut risque.

Le pipeline utilise un `ColumnTransformer` pour normaliser les données numériques et encoder les variables catégorielles, tout en préservant le format DataFrame pour l'étape d'interprétabilité.

---

##  3. Modélisation et Performance
Nous avons opté pour un modèle **XGBoost** optimisé par recherche aléatoire d'hyperparamètres, avec une gestion stricte du déséquilibre des classes.

### Résultats de l'Optimisation :
- **Meilleurs Paramètres :** `{'subsample': 0.8, 'reg_lambda': 10, 'reg_alpha': 1, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 0, 'colsample_bytree': 0.9}`
- **ROC-AUC :** **0.8484**
- **Recall (Capture du Churn) :** **0.8583** (Nous identifions 86% des clients qui vont réellement partir).
- **F1-Score :** 0.6120

### Matrice de Confusion (Seuil 0.4) :
| | Prédit Restant | Prédit Partant |
|---|---|---|
| **Réel Restant** | 681 | 354 |
| **Réel Partant** | 53 | **321** |

---

##  4. Analyse d'Interprétabilité (SHAP)
L'utilisation de **SHAP** permet de justifier chaque prédiction auprès des équipes métier :
- **Contrat Month-to-month :** Facteur n°1 de risque.
- **Tenure :** Plus elle est faible, plus l'impact sur le churn est positif.
- **Monthly Charges :** Un coût mensuel élevé sans services de support associés est un signal d'alerte critique.

---

##  5. Simulateur d'Impact Business
Le modèle ne se contente pas de prédire, il aide à décider. Notre simulateur de ROI permet de définir le seuil d'intervention optimal.

**Résultats de la Simulation :**
En réglant le modèle sur un seuil de **0.47** :
- **Identification :** 309 vrais départs potentiels détectés.
- **Maîtrise :** 306 fausses alertes (coût de rétention maîtrisé).
- **Impact Financier :** **GAIN NET POUR L'ENTREPRISE : 77,550.00 €** sur le segment de test.

---

##  Recommandations Stratégiques
1.  **Migration de Contrat :** Inciter agressivement les clients "Month-to-month" à passer sur des contrats d'un an via des remises ciblées sur les 3 premiers mois.
2.  **Focus Fibre Optique :** Offrir le "Tech Support" gratuitement pendant 6 mois pour tout nouvel abonnement fibre afin de stabiliser la base client.
3.  **Onboarding Critique :** Déclencher une campagne de "Customer Success" automatique pour tous les clients ayant une tenure < 6 mois et des factures > 70$.
