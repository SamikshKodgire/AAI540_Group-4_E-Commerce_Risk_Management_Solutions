import subprocess
import sys

# Force the container to install the SageMaker SDK before doing anything else
print("Installing sagemaker SDK...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker<3.0.0", "-q"])
print("Installation complete.")

import os
import pandas as pd
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

def main():
    print("Starting Feature Store Ingestion Job...")
    
    # The pipeline will map the S3 output from step_process to this folder
    input_dir = '/opt/ml/processing/input'
    
    # Load the intact CSV that we saved in the previous step
    df = pd.read_csv(os.path.join(input_dir, 'final_features.csv'))
    
    # Initialize AWS sessions
    region = boto3.Session().region_name
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client('sagemaker', region_name=region)
    featurestore_runtime = boto_session.client('sagemaker-featurestore-runtime', region_name=region)
    
    feature_store_session = sagemaker.Session(
        boto_session=boto_session, 
        sagemaker_client=sagemaker_client, 
        sagemaker_featurestore_runtime_client=featurestore_runtime
    )
    
    # Connect and Ingest
    feature_group_name = "olist-customer-risk-features-v2" 
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)

    # Pass the dataframe so the FeatureGroup object maps the schema!
    print("Loading feature group metadata...")
    feature_group.load_feature_definitions(data_frame=df) 
    
    print(f"Ingesting {len(df)} records into {feature_group_name}...")
    feature_group.ingest(data_frame=df, max_workers=3, wait=True)
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
