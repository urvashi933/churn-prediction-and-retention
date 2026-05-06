import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Premium Visual Aesthetics Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
MODERN_BLUE = "#2C3E50"
CHURN_RED = "#E74C3C"
STAY_BLUE = "#3498DB"
NEUTRAL_GREY = "#95A5A6"

def setup_plot_style():
    """Applies premium styling to matplotlib/seaborn."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.labelcolor'] = MODERN_BLUE
    plt.rcParams['axes.edgecolor'] = NEUTRAL_GREY
    plt.rcParams['xtick.color'] = MODERN_BLUE
    plt.rcParams['ytick.color'] = MODERN_BLUE

def main():
    """
    Main execution pipeline for Customer Churn Prediction.
    This script handles the full lifecycle from data loading to risk profiling 
    to provide actionable business insights for retention strategies.
    """
    setup_plot_style()
    
    # 1. Environment Setup
    os.makedirs('images', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # 2. Data Acquisition
    print("Loading dataset...")
    df = pd.read_csv('dataset/part_3_customer_churn_prediction.csv')
    
    # 3. Data Cleaning and Preprocessing
    print("Preprocessing data...")
    
    # Handling potential non-numeric strings in TotalCharges (common issue)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Imputing missing values in TotalCharges. 
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Target Encoding: Converting the 'Churn' variable into a binary numeric format (1=Yes, 0=No).
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 4. Exploratory Data Analysis (EDA) - Visualizing Business Patterns
    print("Generating EDA plots for business insights...")
    
    # Pattern 1: Market Share of Churn
    plt.figure(figsize=(7,7))
    plt.pie(df['Churn_Numeric'].value_counts(), 
            labels=['Loyal Customers', 'Churned'], 
            autopct='%1.1f%%', 
            colors=[STAY_BLUE, CHURN_RED], 
            startangle=140, 
            explode=(0, 0.1), 
            shadow=True,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Distribution of Customer Churn', fontsize=15, pad=20, color=MODERN_BLUE)
    plt.savefig('images/overall_churn_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pattern 2: Impact of Contract Commitment
    plt.figure(figsize=(10,6))
    sns.countplot(x='Contract', hue='Churn', data=df, palette=[STAY_BLUE, CHURN_RED])
    plt.title('Churn Performance by Contract Type', fontsize=14, pad=15)
    plt.xlabel('Contract Duration', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('images/churn_by_contract.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pattern 3: Tenure and Customer Loyalty
    plt.figure(figsize=(10,6))
    sns.kdeplot(df[df['Churn'] == 'No']['Tenure'], fill=True, color=STAY_BLUE, label='Stayed', alpha=0.5)
    sns.kdeplot(df[df['Churn'] == 'Yes']['Tenure'], fill=True, color=CHURN_RED, label='Churned', alpha=0.5)
    plt.title('Tenure Distribution: Density Analysis', fontsize=14)
    plt.xlabel('Months of Tenure', fontsize=12)
    plt.legend()
    plt.savefig('images/churn_by_tenure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pattern 4: Price Sensitivity Analysis
    plt.figure(figsize=(10,6))
    sns.violinplot(x='Churn', y='MonthlyCharges', data=df, palette=[STAY_BLUE, CHURN_RED], inner="quartile")
    plt.title('Price Sensitivity: Monthly Charges Distribution', fontsize=14)
    plt.xticks([0, 1], ['Stayed', 'Churned'])
    plt.savefig('images/churn_by_monthly_charges.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pattern 5: Friction in Payment Methods
    plt.figure(figsize=(12,6))
    sns.countplot(x='PaymentMethod', hue='Churn', data=df, palette=[STAY_BLUE, CHURN_RED])
    plt.title('Churn Analysis by Payment Method', fontsize=14)
    plt.xticks(rotation=15)
    plt.savefig('images/churn_by_payment_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pattern 6: Service Portfolio Performance
    plt.figure(figsize=(10,6))
    sns.countplot(x='InternetService', hue='Churn', data=df, palette=[STAY_BLUE, CHURN_RED])
    plt.title('Retention Performance by Internet Service Type', fontsize=14)
    plt.savefig('images/churn_by_internet_service.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pattern 7: Senior Citizen Engagement
    plt.figure(figsize=(8,6))
    sns.barplot(x='SeniorCitizen', y='Churn_Numeric', data=df, palette='coolwarm', ci=None)
    plt.title('Churn Rate: Senior Citizens vs General Population', fontsize=14)
    plt.xticks([0, 1], ['Non-Senior', 'Senior Citizen'])
    plt.ylabel('Churn Probability')
    plt.savefig('images/churn_by_senior_citizen.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pattern 8: Holistic Risk Visualization
    plt.figure(figsize=(10,7))
    sns.scatterplot(x='Tenure', y='MonthlyCharges', hue='Churn', data=df, 
                    palette=[STAY_BLUE, CHURN_RED], alpha=0.6, s=60, edgecolor='w')
    plt.title('Risk Landscape: Tenure vs Billing Intensity', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('images/tenure_charges_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Feature Engineering for Predictive Modeling
    df_model = df.copy()
    customer_ids = df_model.pop('CustomerID')
    df_model.drop(['Churn'], axis=1, inplace=True) # Dropping original categorical target
    
    # One-hot encoding categorical features.
    categorical_cols = df_model.select_dtypes(include=['object']).columns
    df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    # Splitting into Training and Testing sets.
    X = df_model.drop('Churn_Numeric', axis=1)
    y = df_model['Churn_Numeric']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling numerical features.
    numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # 6. Predictive Modeling
    print("\nDeveloping Predictive Models...")
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            'model': model,
            'preds': preds,
            'acc': accuracy_score(y_test, preds),
            'prec': precision_score(y_test, preds),
            'rec': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'cm': confusion_matrix(y_test, preds)
        }
    
    # 7. Model Performance Evaluation
    for name, metrics in results.items():
        print(f"\n--- {name} Business KPIs ---")
        print(f"Accuracy:   {metrics['acc']:.4f} | Precision: {metrics['prec']:.4f} | Recall: {metrics['rec']:.4f} | F1: {metrics['f1']:.4f}")
        
        plt.figure(figsize=(6,5))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'],
                    annot_kws={"size": 14, "weight": "bold"})
        plt.title(f'Confusion Matrix: {name}', fontsize=14, pad=15)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'images/cm_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # 8. Churn Risk Profiling (Using Logistic Regression for high interpretability)
    print("\nGenerating Actionable Risk Report...")
    lr_model = results['Logistic Regression']['model']
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
    
    # Reorganizing columns for clarity
    cols = ['CustomerID', 'Risk_Level', 'Churn_Probability'] + [c for c in results_df.columns if c not in ['CustomerID', 'Risk_Level', 'Churn_Probability']]
    results_df = results_df[cols]
    results_df.to_csv('outputs/churn_risk_predictions.csv', index=False)
    
    print(f"Risk Profile Distribution:\n{results_df['Risk_Level'].value_counts()}")
    print("\nPipeline execution complete. All business assets saved to 'images/' and 'outputs/'.")

if __name__ == "__main__":
    main()
