# Rural Texas Healthcare Access Analysis

## Project Overview
This repository contains code and analysis for identifying and characterizing population segments with distinct healthcare access patterns in rural Texas. Using clustering techniques, we identify meaningful groups based on health needs, social determinants of health (SDOH), and healthcare access factors.

## Data Sources
The analysis uses survey data from rural Texas residents, collected through:
- RTPH_Survey_stand_TLL_4.1.25.csv
- NUMERIC_HealthIssuesRuralTX_Net.csv

## Repository Structure
- `wic_data_engineering.ipynb`: Initial data preprocessing, feature engineering, and dataset preparation
- `clustering_analysis.ipynb`: Implementation of K-means clustering with different feature sets, evaluation, and interpretation

## Methodology

### Data Engineering and Preparation

#### Feature Engineering
- **Age Categorization**: Dividing respondents into age groups (18-44, 45-64, 65+)
- **Health Burden Quantification**: Creating `health_issues_count` by summing indicators for poor physical health, poor mental health, poor quality of life, and chronic conditions
- **SDOH Vulnerability Score**: Composite measure based on food, housing, transportation, and income insecurity
- **Healthcare Proximity Measures**: 
  - `facilities_within_30min`: Count of healthcare facilities within 30 minutes
  - `facilities_within_10miles`: Count of healthcare facilities within 10 miles
  - Binary indicators for specific facility types nearby (primary care, urgent care, ER)
- **Incorrect Care Indicator**: Flag for instances where respondents reported using inappropriate level of care

#### Data Preprocessing
- Selection of relevant variables for clustering analysis
- Missing data handling with context-appropriate imputation
- Standardization of continuous variables using StandardScaler
- One-hot encoding of categorical variables

### Clustering Analysis

#### Feature Set Approaches
1. **Individual Variables**: Uses discrete individual features related to demographics, health, social determinants, and access
2. **Composite Variables**: Utilizes composite scores (health_issues_count, sdoh_vulnerability_score) alongside individual features
3. **Balanced Approach**: Integration of both individual indicators and composite measures

#### Optimal Cluster Determination
- Silhouette score analysis to identify optimal number of clusters for each feature set
- Automated process for cluster number optimization using `find_optimal_clusters` function

#### Cluster Evaluation
- Inter-approach agreement calculation
- Variance explained in key outcome variables by cluster assignment
- Comparison across clustering approaches to identify the most effective one

#### Detailed Profiling
- Statistical profiling of each cluster
- Identification of representative and extreme cases
- Visualizations of cluster characteristics through heatmaps and bar plots

## Key Findings
The analysis identifies distinct population segments with unique patterns of healthcare needs, barriers, and utilization. These segments can be targeted for specific interventions to improve healthcare access and outcomes in rural Texas.

## Usage
1. Run `wic_data_engineering.ipynb` to preprocess the data and generate the required datasets
2. Execute `clustering_analysis.ipynb` to perform clustering, evaluate results, and generate insights

## Dependencies
- Python 3.11
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Future Work
- Development of targeted intervention strategies for each identified segment
- Validation of clusters with additional data sources
- Geographic analysis of cluster distribution across rural Texas

## Contact
For questions about this project, please contact [adhsaksham27@gmail.com].
