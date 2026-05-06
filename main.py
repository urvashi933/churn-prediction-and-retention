import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    """
    Main execution pipeline for Customer Churn Prediction.
    This script handles the full lifecycle from data loading to risk profiling 
    to provide actionable business insights for retention strategies.
    """
    
    # 1. Environment Setup
    # Creating directories to organize visual assets and data outputs for business reporting.
    os.makedirs('images', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # 2. Data Acquisition
    # Loading the raw customer dataset for analysis.
    print("Loading dataset...")
    df = pd.read_csv('dataset/part_3_customer_churn_prediction.csv')
    
    # 3. Data Cleaning and Preprocessing (Critical for high-quality analysis)
    print("Preprocessing data...")
    
    # Imputing missing values in TotalCharges. 
    # Business Rationale: Missing financial data can bias model results; median is used to minimize outlier influence.
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Target Encoding: Converting the 'Churn' variable into a binary numeric format (1=Yes, 0=No).
    # This allows mathematical models to calculate the probability of a customer leaving.
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 4. Exploratory Data Analysis (EDA) - Visualizing Business Patterns
    print("Generating EDA plots for business insights...")
    
    # Pattern 1: Market Share of Churn
    plt.figure(figsize=(6,6))
    df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Churn', 'Churn'], colors=['skyblue', 'salmon'])
    plt.title('Market Share of Customer Churn')
    plt.ylabel('')
    plt.savefig('images/overall_churn_rate.png')
    plt.close()
    
    # Pattern 2: Impact of Contract Commitment on Retention
    plt.figure(figsize=(8,5))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='viridis')
    plt.title('Retention Performance by Contract Type')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.savefig('images/churn_by_contract.png')
    plt.close()
    
    # Pattern 3: Tenure and Customer Loyalty
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Churn', y='Tenure', data=df, palette='Set2')
    plt.title('Customer Loyalty Duration (Tenure) vs Churn')
    plt.xticks([0, 1], ['Stayed', 'Churned'])
    plt.savefig('images/churn_by_tenure.png')
    plt.close()
    
    # Pattern 4: Price Sensitivity Analysis
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
    plt.title('Price Sensitivity Analysis: Monthly Charges vs Churn')
    plt.xticks([0, 1], ['Stayed', 'Churned'])
    plt.savefig('images/churn_by_monthly_charges.png')
    plt.close()
    
    # Pattern 5: Friction in Payment Methods
    plt.figure(figsize=(10,5))
    sns.countplot(x='PaymentMethod', hue='Churn', data=df, palette='viridis')
    plt.title('Customer Churn Analysis by Payment Method')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/churn_by_payment_method.png')
    plt.close()
    
    # Pattern 6: Service Portfolio Performance
    plt.figure(figsize=(8,5))
    sns.countplot(x='InternetService', hue='Churn', data=df, palette='viridis')
    plt.title('Service Portfolio Retention Performance')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.savefig('images/churn_by_internet_service.png')
    plt.close()
    
    # Pattern 7: Senior Citizen Engagement
    plt.figure(figsize=(6,5))
    sns.countplot(x='SeniorCitizen', hue='Churn', data=df, palette='viridis')
    plt.title('Churn Profile: Senior Citizens vs General Population')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.xticks([0, 1], ['Non-Senior', 'Senior Citizen'])
    plt.savefig('images/churn_by_senior_citizen.png')
    plt.close()
    
    # Pattern 8: Holistic Risk Visualization
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Tenure', y='MonthlyCharges', hue='Churn', data=df, palette='coolwarm', alpha=0.6)
    plt.title('Risk Landscape: Relationship between Tenure, Billing, and Churn')
    plt.savefig('images/tenure_charges_churn.png')
    plt.close()
    
    # 5. Feature Engineering for Predictive Modeling
    # Dropping non-predictive CustomerID for modeling but preserving for the final report.
    df_model = df.copy()
    customer_ids = df_model.pop('CustomerID')
    
    # One-hot encoding categorical features.
    categorical_cols = df_model.select_dtypes(include=['object']).columns
    df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    # Splitting into Training and Testing sets.
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling numerical features.
    numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # 6. Predictive Modeling
    print("\nDeveloping Predictive Models...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # 7. Model Performance Evaluation
    def evaluate_model(y_true, y_pred, model_name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n--- {model_name} Business KPIs ---")
        print(f"Accuracy:   {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
        plt.title(f'Outcome Matrix: {model_name}')
        plt.savefig(f'images/cm_{model_name.replace(" ", "_").lower()}.png')
        plt.close()
        
    evaluate_model(y_test, lr_preds, "Logistic Regression")
    evaluate_model(y_test, rf_preds, "Random Forest")
    
    # 8. Churn Risk Profiling
    print("\nGenerating Actionable Risk Report...")
    risk_probs = lr_model.predict_proba(X_test)[:, 1]
    
    X_test_unscaled = X_test.copy()
    X_test_unscaled[numerical_cols] = scaler.inverse_transform(X_test[numerical_cols])
    
    results_df = X_test_unscaled.copy()
    results_df['CustomerID'] = customer_ids.loc[X_test.index].values
    results_df['Churn_Probability'] = risk_probs
    
    def assign_risk(prob):
        if prob > 0.7: return 'High Risk'
        elif prob > 0.4: return 'Medium Risk'
        else: return 'Low Risk'
            
    results_df['Risk_Level'] = results_df['Churn_Probability'].apply(assign_risk)
    
    # Sorting by risk for the Business Analyst.
    results_df = results_df[['CustomerID', 'Risk_Level', 'Churn_Probability'] + [c for c in results_df.columns if c not in ['CustomerID', 'Risk_Level', 'Churn_Probability']]]
    results_df.to_csv('outputs/churn_risk_predictions.csv', index=False)
    
    print(f"Risk Profile Distribution:\n{results_df['Risk_Level'].value_counts()}")
    print("\nExecution Successful! Reports ready in 'images/' and 'outputs/'.")

if __name__ == "__main__":
    main()
