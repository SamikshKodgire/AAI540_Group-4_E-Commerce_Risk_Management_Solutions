# **E-Commerce Risk Management Solutions**

**Author:**
  Manoj Nair, 
  
  Samiksha Kodgire

**Project Overview**

This project focuses on building an end-to-end MLOps-based machine learning system capable of predicting high-risk ecommerce transactions. Ecommerce marketplaces such as Amazon, Flipkart, and Myntra often face operational and financial losses due to product returns, cancelled orders, delayed deliveries, and poor customer experiences. These issues increase logistics costs, negatively impact seller performance, and reduce customer satisfaction.
The objective of this project is to proactively identify high-risk ecommerce orders using machine learning so that businesses can take preventive operational actions such as improving delivery prioritisation, strengthening customer communication, or reviewing seller performance.
The project simulates a real-world production ML workflow and incorporates multiple MLOps concepts including:

  1. Data ingestion and preprocessing pipelines
  2. Feature engineering workflows
  3. Experiment tracking with MLflow
  4. Model training and evaluation
  5. Deployment using AWS SageMaker
  6. Batch inference pipelines
  7. Model monitoring and drift detection
  8. CI/CD integration workflows

     
**Problem Statement**

The goal of this project is to predict whether an ecommerce order is likely to become a high-risk transaction based on customer behaviour, payment information, seller details, delivery timelines, and review outcomes.

The project is implemented as a Binary Classification machine learning problem where:

1 = High-Risk Order
0 = Low-Risk Order

The target variable is engineered using indicators such as:

  Low review scores
  Cancelled or unavailable orders
  Delivery delays
  Customer dissatisfaction signals
  Dataset
  Dataset Used


**Brazilian E-Commerce Public Dataset by Olist**
_Dataset Link: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce_

**Key Dataset Tables**
1. Customers
2. Orders
3. Order Items
4. Payments
5. Reviews
6. Products
7. Sellers
8. Geolocation
The dataset contains approximately 100,000 real-world ecommerce transactions across multiple relational tables.


**Technologies Used**
  Programming Language
  Python
  Libraries
  Pandas
  NumPy
  Scikit-learn
  XGBoost
  Matplotlib
  Seaborn
  MLflow
  Cloud & MLOps Tools
  AWS SageMaker
  Amazon S3
  GitHub Actions
  MLflow
  Proposed Workflow
  Data ingestion and relational joins
  Data cleaning and preprocessing
  Feature engineering
  Exploratory data analysis
  Model training and evaluation
  Experiment tracking
  Model deployment
  Monitoring and drift detection
  CI/CD pipeline integration
  Evaluation Metrics


The project will primarily evaluate model performance using:**
**
  Recall
  F1-Score
  Precision
  ROC-AUC Score

Recall is prioritised because failing to identify a risky transaction could result in customer dissatisfaction, operational losses, and return-related costs.


**Project Goals**

  Build a production-ready ML pipeline for ecommerce risk prediction
  Implement MLOps workflows using AWS SageMaker
  Deploy the trained model as an inference endpoint
  Implement monitoring and drift detection
  Demonstrate CI/CD concepts and reproducible ML workflows
  
**Non-Goals**

  Deep learning architectures
  Real-time streaming systems
  Full frontend application development
  Recommendation systems
  NLP-heavy review sentiment analysis

  

**Potential future enhancements include:**

1. Real-time inference pipelines
2. Advanced drift monitoring
3. Explainable AI dashboards
4. Seller risk scoring systems
5. Customer segmentation
6. Automated retraining workflows

