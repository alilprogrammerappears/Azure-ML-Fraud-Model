import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from azureml.core import Workspace, Dataset


def train_model(x, y):

    mlflow.start_run()

    # split data into testing and training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize and process the data
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Model training
    print("Training...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(x_train_scaled, y_train)

    # results prediction
    y_pred = rf_model.predict(x_test_scaled)
    y_pred_probability = rf_model.predict_proba(x_test_scaled)[:, 1]

    # Metrics calculations
    print("Calculating results")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_probability)
    f1 = f1_score(y_test, y_pred)

    # Format results
    results = []
    name = 'Random Forest Model'
    
    results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC-ROC': auc_roc,
        })
    
    # Register and save the model
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=rf_model,
        registered_model_name='rf_model',
        artifact_path='rf_model',
    )

    mlflow.sklearn.save_model(
        sk_model=rf_model,
        path=os.path.join('rf_model', "trained_rf_model"),
    )

    return results

# get the dataset from Azure file
ws = Workspace.from_config()
dataset_name = "fraud_dataset"
dataset = Dataset.get_by_name(ws, dataset_name)
df = dataset.to_pandas_dataframe()

x = df.drop(['id', 'Class'], axis=1)
y = df['Class']

results = train_model(x, y)

for key, value in results[0].items():
    print(f"{key}: {value}")

mlflow.end_run()
