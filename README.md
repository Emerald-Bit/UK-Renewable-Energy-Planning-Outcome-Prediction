# Predicting UK Renewable Energy Planning Outcomes

**Stack:** PyTorch, MLFlow, Pandas, Numpy, SkiKitLearn

**Data used:** REPD_Publication_Q3_2025.csv

**Data Source:** https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract

**Final Outcomes:**
_Macro F1_
- Logistic Regression (Baseline):   0.6042 
- Neural Network (Optimised):       0.4355


My interest in this project was deeply personal from the outset. As a local to the Park Royal and Old Oak Common area, I have had a front-row seat to the rapid urban regeneration taking place on my own doorstep. One example that particularly stood out to me was the local district heating network, which was designed to capture waste heat from a newly built data centre and use it to sustainably heat nearby homes.

Projects like this show how local planning authorities and development corporations, such as the OPDC, are increasingly placing green infrastructure, circular-economy hubs, and sustainable regeneration at the centre of future development. This is not just about modernising places. It is about meeting the UK’s net-zero ambitions in a way that is practical, local, and economically meaningful. Yet for all the promise of renewable energy projects, whether district heating, solar, wind, or battery storage, one major bottleneck remains: the statutory planning process.

Securing planning permission for renewable energy infrastructure is often far more complex than it first appears. Fragmented land ownership, local policy constraints, and legitimate community concerns can quickly turn an otherwise viable project into a costly and time-consuming challenge. When an application is refused, the impact goes well beyond paperwork. It can mean lost capital, delayed decarbonisation, and missed opportunities for local green-collar job creation. That is why understanding the less obvious patterns behind which projects are approved and which are refused matters so much. It has real value for developers, councils, and policymakers trying to de-risk investment decisions and accelerate sustainable development.

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
