# Predicting UK Renewable Energy Planning Outcomes

**Stack:** PyTorch, MLFlow, Pandas, Numpy, SkiKitLearn

**Data used:** REPD_Publication_Q3_2025.csv

**Data Source:** https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract

**Final Outcomes:**
_Macro F1_
- Logistic Regression (Baseline):   0.6042 
- Neural Network (Optimised):       0.4355


As a local to the Park Royal and Old Oak Common area, my initial interest in this project was sparked by observing the rapid urban regeneration happening on my own doorstep. A prime example of this is the local district heating network, which was intended to capture waste heat from a newly built data centre to sustainably warm nearby homes.

Ambitious projects like this highlight how local planning authorities and development corporations (such as the OPDC) are increasingly prioritising green infrastructure, circular-economy hubs, and sustainable regeneration to meet the UK’s net-zero targets. However, deploying renewable energy projects, whether district heating, solar, wind, or battery storage, faces a significant bottleneck: the statutory planning process.

Navigating fragmented land ownership, community concerns, and stringent local regulations makes securing planning permission complex and expensive. A refused application represents a massive loss of time, capital, and potential 'green-collar' job creation. Understanding the hidden patterns behind which projects get approved and which get refused is critical for developers, local councils, and policymakers looking to de-risk investments and accelerate sustainable development.

Project Objectives

This project aims to reverse-engineer the UK's renewable energy planning decisions using Machine Learning. By analysing historical application data, including geographic regions, technology types, installed capacities, and NLP embeddings of site text. This notebook develops a predictive pipeline to classify whether a proposed project will be Approved or Refused.

Specifically, this project seeks to:
- Identify High-Risk Applications: Build a model capable of flagging likely 'Refusals' before millions of pounds are spent on development.
- Solve Severe Data Imbalance: Tackle the inherent 89:11 class imbalance in planning data (where approvals heavily outweigh refusals) by engineering custom loss functions and decision thresholds.
- Establish a Scientific Benchmark: Rigorously compare the performance of complex Deep Learning architectures (PyTorch Neural Networks) against highly interpretable, traditional baselines (Logistic Regression) to determine the most efficient approach for this specific tabular dataset.

Key Technical Highlights
- End-to-End Pipeline: Data cleaning, robust leakage prevention, and advanced imputation strategies.
- Deep Learning (PyTorch): Implementation of a custom Neural Network, bypassing standard activation functions in favour of BCEWithLogitsLoss for mathematical stability.
- Advanced Imbalance Handling: Utilisation of dynamic class weighting (optimising for the minority 'Refused' class) combined with post-training probability threshold tuning to maximise the Macro F1 Score.
- NLP Integration: Incorporating text embeddings to capture the semantic nuances of operator and site names.
