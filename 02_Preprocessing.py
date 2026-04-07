import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class ChurnExpertPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.preprocessor = None

    def _feature_engineering(self, df):
        """
        Crée des variables à haute valeur prédictive identifiées en EDA.
        """
        # 1. TotalServices : Somme des options souscrites (Stickiness)
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['TotalServices'] = (df[services] == 'Yes').sum(axis=1)
        
        # 2. AvgMonthlyCharges : Ratio historique pour détecter les anomalies de facturation
        # On évite la division par zéro pour les nouveaux clients (tenure=0)
        df['ChargePerMonth_Ratio'] = df['TotalCharges'] / (df['tenure'].replace(0, 1))
        
        # 3. IsFiber : Isoler le segment à risque identifié en EDA
        df['IsFiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
        
        return df

    def load_and_prepare(self):
        df = pd.read_csv(self.data_path)
        
        # Nettoyage strict
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
        # Drop ID
        df = df.drop(columns=['customerID']) if 'customerID' in df.columns else df
        
        # Application du Feature Engineering
        df = self._feature_engineering(df)
        
        return df

    def get_pipeline(self):
        """
        Configure un pipeline qui sépare bien le scaling du OneHot.
        """
        # Mise à jour des listes avec les nouvelles features
        num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'ChargePerMonth_Ratio']
        cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaperlessBilling', 'PaymentMethod']

        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
        
        return self.preprocessor

    def process_to_df(self, X_train, X_test):
        """
        Transforme les données tout en gardant un format DataFrame Pandas.
        CRUCIAL pour l'interprétabilité SHAP plus tard.
        """
        self.get_pipeline()
        
        # Fit sur Train, Transform sur les deux
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)
        
        # Récupération propre des noms de colonnes
        cat_cols = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out()
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'ChargePerMonth_Ratio']
        all_cols = list(num_cols) + list(cat_cols)
        
        # Reconstruction des DataFrames
        X_train_final = pd.DataFrame(X_train_proc, columns=all_cols, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_proc, columns=all_cols, index=X_test.index)
        
        return X_train_final, X_test_final

if __name__ == "__main__":
    pipe_manager = ChurnExpertPipeline('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_full = pipe_manager.load_and_prepare()
    
    X = df_full.drop('Churn', axis=1)
    y = (df_full['Churn'] == 'Yes').astype(int)
    
    # Split stratifié pour respecter l'équilibre des classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Transformation
    X_train_final, X_test_final = pipe_manager.process_to_df(X_train, X_test)
    
    print(f"Preprocessing Expert terminé.")
    print(f"Nouvelles colonnes créées : {[c for c in X_train_final.columns if 'TotalServices' in c or 'Ratio' in c]}")
    print(f"Shape finale : {X_train_final.shape}")