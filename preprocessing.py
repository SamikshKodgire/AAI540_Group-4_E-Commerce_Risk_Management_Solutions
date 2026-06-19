import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    # SageMaker Processing standard directories
    base_dir = '/opt/ml/processing'
    input_dir = os.path.join(base_dir, 'input')
    
    print("Loading datasets from S3 input...")
    # Assuming standard Olist CSV files are passed to the Processing job
    orders = pd.read_csv(os.path.join(input_dir, 'olist_orders_dataset.csv'))
    payments = pd.read_csv(os.path.join(input_dir, 'olist_order_payments_dataset.csv'))
    reviews = pd.read_csv(os.path.join(input_dir, 'olist_order_reviews_dataset.csv'))
    customers = pd.read_csv(os.path.join(input_dir, 'olist_customers_dataset.csv'))
    
    print("Merging relational tables...")
    # Join tables based on relational keys
    df = orders.merge(customers, on='customer_id', how='left')
    df = df.merge(payments, on='order_id', how='left')
    df = df.merge(reviews, on='order_id', how='left')
    
    print("Executing feature engineering and handling nulls...")
    # Parse datetimes and handle missing values
    datetime_cols = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    # --- TARGET LABEL GENERATION ---
    # Business Logic: High Risk (1) if review score <= 2 AND a logistical failure occurred
    df['logistical_failure'] = (
        (df['order_status'] != 'delivered') | 
        (df['order_delivered_customer_date'] > df['order_estimated_delivery_date'])
    )
    df['is_high_risk'] = np.where((df['review_score'] <= 2) & df['logistical_failure'], 1, 0)
    
    # --- FEATURE ENGINEERING ---
    # Calculate delivery delay in days (negative means early delivery)
    df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['delivery_delay_days'] = df['delivery_delay_days'].fillna(0) # Default to 0 for un-delivered/missing
    
    df['payment_value'] = df['payment_value'].fillna(0)
    
    # Select features defined in the ML system design
    selected_features = [
        'payment_type', 
        'payment_installments', 
        'payment_value', 
        'customer_state', 
        'delivery_delay_days', 
        'is_high_risk'
    ]
    
    # Filter dataset and drop duplicate rows resulting from the joins
    df_model = df[selected_features].copy().drop_duplicates()
    df_model = df_model.dropna() # Drop any remaining stray NaNs to ensure XGBoost compatibility
    
    print("Encoding categorical variables...")
    # XGBoost requires numeric inputs; encode categoricals
    le = LabelEncoder()
    df_model['payment_type'] = le.fit_transform(df_model['payment_type'].astype(str))
    df_model['customer_state'] = le.fit_transform(df_model['customer_state'].astype(str))
        
    print("Splitting data into Train and Test sets...")
    # Train-test split (80/20) with stratification on the imbalanced target variable
    train_df, test_df = train_test_split(
        df_model, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_model['is_high_risk']
    )
    
    # SageMaker Built-in XGBoost expects the target variable to be the first column
    cols = ['is_high_risk'] + [col for col in train_df.columns if col != 'is_high_risk']
    train_df = train_df[cols]
    test_df = test_df[cols]
    
    print("Saving processed data to outputs...")
    # Paths configured to match the ProcessingOutput destinations in the Pipeline
    train_output_path = os.path.join(base_dir, 'train', 'train.csv')
    test_output_path = os.path.join(base_dir, 'test', 'test.csv')
    
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    
    # Save without headers and indices, as required by SageMaker XGBoost
    train_df.to_csv(train_output_path, header=False, index=False)
    test_df.to_csv(test_output_path, header=False, index=False)
    
    print("Data Engineering step completed successfully.")

if __name__ == "__main__":
    main()
