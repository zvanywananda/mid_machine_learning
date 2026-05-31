import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("Preprocessed Balanced dataset.csv") 
 
kolom_contekan = ['Label', 'Attack_Category', 'Attack_sub_category']
X = df.drop(columns=[col for col in kolom_contekan if col in df.columns])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

best_pipeline = joblib.load('pipeline_terbaik.pkl')
le = joblib.load('label_encoder.pkl')
y_test_encoded = le.transform(y_test)

mlflow.set_experiment("IoT_Vulnerability_Detection")

with mlflow.start_run(run_name="Best_Pipeline_Model"):

    mlflow.log_param("Feature Selection", "SelectKBest")
    mlflow.log_param("K_Features", best_pipeline.named_steps['feature_selection'].k)
    mlflow.log_param("n_estimators", best_pipeline.named_steps['model'].n_estimators)
    
    y_pred = best_pipeline.predict(X_test)

    acc = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average='macro')
    
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("F1_Macro", f1)
    
    mlflow.sklearn.log_model(best_pipeline, "best_pipeline_model")
    
    print(f"Tracking kelar! Accuracy: {acc:.4f}, F1: {f1:.4f}")