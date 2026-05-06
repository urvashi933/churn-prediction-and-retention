import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    # 1. Create output directory
    os.makedirs('images', exist_ok=True)
    
    # 2. Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('dataset/part_3_customer_churn_prediction.csv')
    
    # 3. Data Cleaning and Preprocessing
    print("Preprocessing data...")
    # Handling missing values in TotalCharges (imputing with median)
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Target encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 4. Exploratory Data Analysis
    print("Generating EDA plots...")
    
    # Overall churn rate
    plt.figure(figsize=(6,6))
    df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Churn', 'Churn'], colors=['skyblue', 'salmon'])
    plt.title('Overall Churn Rate')
    plt.ylabel('')
    plt.savefig('images/overall_churn_rate.png')
    plt.close()
    
    # Churn by contract type
    plt.figure(figsize=(8,5))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='viridis')
    plt.title('Churn by Contract Type')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.savefig('images/churn_by_contract.png')
    plt.close()
    
    # Churn by tenure
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Churn', y='Tenure', data=df, palette='Set2')
    plt.title('Churn by Tenure')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.savefig('images/churn_by_tenure.png')
    plt.close()
    
    # Churn by monthly charges
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
    plt.title('Churn by Monthly Charges')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.savefig('images/churn_by_monthly_charges.png')
    plt.close()
    
    # Churn by payment method
    plt.figure(figsize=(10,5))
    sns.countplot(x='PaymentMethod', hue='Churn', data=df, palette='viridis')
    plt.title('Churn by Payment Method')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/churn_by_payment_method.png')
    plt.close()
    
    # Churn by internet service
    plt.figure(figsize=(8,5))
    sns.countplot(x='InternetService', hue='Churn', data=df, palette='viridis')
    plt.title('Churn by Internet Service Type')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.savefig('images/churn_by_internet_service.png')
    plt.close()
    
    # Churn by senior citizen status
    plt.figure(figsize=(6,5))
    sns.countplot(x='SeniorCitizen', hue='Churn', data=df, palette='viridis')
    plt.title('Churn by Senior Citizen Status')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.xticks([0, 1], ['No (0)', 'Yes (1)'])
    plt.savefig('images/churn_by_senior_citizen.png')
    plt.close()
    
    # Relationship between tenure, charges, and churn
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Tenure', y='MonthlyCharges', hue='Churn', data=df, palette='coolwarm', alpha=0.6)
    plt.title('Relationship between Tenure, Monthly Charges, and Churn')
    plt.savefig('images/tenure_charges_churn.png')
    plt.close()
    
    # 5. Feature Engineering for Modeling
    # Drop CustomerID as it's not a predictive feature
    df_model = df.drop('CustomerID', axis=1)
    
    # Identify categorical columns
    categorical_cols = df_model.select_dtypes(include=['object']).columns
    
    # One-hot encoding
    df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    # Train-test split
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling numerical columns
    numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # 6. Model Building
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # 7. Model Evaluation
    def evaluate_model(y_true, y_pred, model_name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n--- {model_name} Evaluation ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'images/cm_{model_name.replace(" ", "_").lower()}.png')
        plt.close()
        
    evaluate_model(y_test, lr_preds, "Logistic Regression")
    evaluate_model(y_test, rf_preds, "Random Forest")
    
    # 8. Churn Risk Interpretation (using Logistic Regression for probabilities)
    print("\nCalculating Churn Risk Profiles...")
    risk_probs = lr_model.predict_proba(X_test)[:, 1]
    
    X_test_unscaled = X_test.copy()
    X_test_unscaled[numerical_cols] = scaler.inverse_transform(X_test[numerical_cols])
    
    results_df = X_test_unscaled.copy()
    results_df['Churn_Probability'] = risk_probs
    
    def assign_risk(prob):
        if prob > 0.7:
            return 'High Risk'
        elif prob > 0.4:
            return 'Medium Risk'
        else:
            return 'Low Risk'
            
    results_df['Risk_Level'] = results_df['Churn_Probability'].apply(assign_risk)
    
    print("\nRisk Level Distribution in Test Set:")
    print(results_df['Risk_Level'].value_counts())
    
    # 9. Save outputs
    os.makedirs('outputs', exist_ok=True)
    results_df.to_csv('outputs/churn_risk_predictions.csv', index=False)
    print("Predictions saved to 'outputs/churn_risk_predictions.csv'")
    
    print("\nDone! EDA plots have been saved to 'images/' and predictions to 'outputs/'.")

if __name__ == "__main__":
    main()
