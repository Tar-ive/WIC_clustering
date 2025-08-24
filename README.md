# Rural Women's Healthcare Access Clustering Analysis

## Research Overview

This repository contains the implementation and analysis code for the research paper **"Exploring rural women's healthcare access through social vulnerability profiles: A cluster analysis of regional survey data in Texas"**. The study uses cluster analysis to identify distinct vulnerability profiles among rural women in East Texas based on health needs, socioeconomic determinants, and geographic access to care.

### Key Research Findings

The study identified **7 distinct vulnerability profiles** among 159 rural women surveyed at WIC service centers in Texas' Public Health Region 4/5N. These profiles revealed how overlapping structural and contextual vulnerability indicators, rather than demographic characteristics alone, shape healthcare access for rural women.

## Dataset

The analysis uses survey data from:
- **159 adult women** identifying as female
- **22 rural and medically underserved counties** in East Texas
- Data collected at WIC service centers
- Variables covering demographics, health status, socioeconomic determinants, geographic access, and healthcare outcomes

## Notebooks Description

### 1. `wic_data_engineering.ipynb`
**Purpose**: Data preparation and variable engineering

This notebook performs the initial data processing and feature engineering required for the clustering analysis. It:
- Maps variable names from the raw survey data to match analysis requirements
- Creates intermediate variables needed for composite measures:
  - Health issues count (combining poor physical/mental health, quality of life, chronic conditions)
  - Social determinants of health (SDOH) vulnerability score
  - Facilities within 30 minutes measure
- Generates binary indicators for high health needs, high social vulnerability, and low geographic access
- Creates the focused dataset with three key dimensions as described in the paper

**Key outputs**:
- `cluster_data_.csv`: Prepared dataset with all engineered features
- `focused_dataset.csv`: Simplified dataset with binary composite variables

### 2. `wic_clustering_analaysis.ipynb`
**Purpose**: Main clustering analysis implementation

This comprehensive notebook implements the full clustering methodology described in the paper. It includes:
- **Multiple clustering approaches**:
  - Individual variables approach (avoiding double-counting)
  - Composite variables approach (using aggregated measures)
  - Balanced approach (combining both)
  - PCA-based dimensionality reduction
  - Hierarchical clustering
- **Clustering algorithms tested**:
  - K-means (primary method, achieving silhouette score of 0.93)
  - Hierarchical clustering (Ward, Complete, Average linkage)
  - Spectral clustering
  - DBSCAN
  - Gaussian Mixture Models
- **Validation metrics**: Silhouette score, Davies-Bouldin Index, Calinski-Harabasz Index
- **Detailed profiling** of each cluster including representative cases
- **Statistical validation** of cluster differences on outcome variables

The notebook reproduces the paper's finding of 7 optimal clusters with excellent separation (silhouette score = 0.93).

### 3. `3_factor .ipynb`
**Purpose**: Three-factor clustering analysis following validated methodology

This notebook implements a focused three-factor clustering approach based on the paper's core dimensions:
1. **High health needs** (chronic conditions or multiple health issues)
2. **High social vulnerability** (2+ socioeconomic barriers)
3. **Low geographic access** (≤2 healthcare facilities within 30 minutes)

The analysis:
- Tests multiple clustering algorithms with standardized evaluation
- Validates the 7-cluster solution as optimal
- Generates comprehensive validation metrics
- Creates visualization of cluster profiles
- Follows the exact methodology described in Lanni et al. (2024)

### 4. `wic_plots.ipynb`
**Purpose**: Visualization and figure generation

This notebook creates publication-quality visualizations including:
- **Demographic distribution plots** across clusters (Figure 4 from the paper)
- Area charts showing the distribution of key characteristics:
  - Person of Color percentage
  - College education rates
  - Employment status
  - Insurance coverage
  - Difficulty affording healthcare
- Individual and combined faceted plots using plotnine (ggplot2-style plotting)
- Cluster profile heatmaps

**Note**: For chart generation, run `3_factor.ipynb` first to generate the necessary cluster data and visualizations.

### 5. `wic_stage4.ipynb`
**Purpose**: Advanced analysis and statistical validation

This notebook performs deeper statistical analysis including:
- **DBSCAN clustering** with optimization for 6-7 clusters
- **Significance testing** of relationships between variables
- **Cross-cutting insights** analysis showing:
  - How barriers drive care avoidance and dissatisfaction
  - The "high needs paradox" (increased care-seeking but lower satisfaction)
  - Importance of outliers/noise points
  - Education and age interactions
- **Statistical validation** using ANOVA, Kruskal-Wallis tests, and regression models
- Generation of comprehensive cluster profiles with outcome analysis

## Key Files Generated

### Data Files
- `cluster_data_.csv`: Full dataset with all engineered features
- `focused_dataset.csv`: Simplified dataset for three-factor analysis
- `final_clustered_data.csv`: Dataset with cluster assignments
- `3factors-2.csv`: Three-factor clustering results

### Visualization Files
- `3d_plot_v2.png`, `3d_plot_v3.png`: 3D scatter plots of clusters
- `3d_scatterplot.png`: Three-dimensional visualization of vulnerability profiles
- `clustering_comparison_grid.png`: Comparison of different clustering methods

## Methodology Highlights

The analysis follows the paper's methodology:

1. **Dimensionality Reduction**: Converting multiple indicators into three binary composite variables
2. **Algorithm Selection**: K-means clustering selected after comparing multiple methods
3. **Validation**: Using silhouette scores, Davies-Bouldin Index, and Calinski-Harabasz Index
4. **Interpretation**: Detailed profiling of each cluster's characteristics and outcomes

## Results Summary

The seven identified clusters represent:
- **Cluster 0** (23.9%): Reference group with no vulnerabilities
- **Cluster 1** (13.8%): High vulnerability (all three indicators present)
- **Clusters 2, 4, 5**: Low vulnerability profiles (one indicator each)
- **Clusters 3, 6**: Moderate vulnerability profiles (two indicators)

These clusters showed significant differences in healthcare access outcomes, with the high vulnerability cluster (Cluster 1) reporting only 73.5% success rate in getting needed care compared to 91.6% for the reference group.

## Requirements

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
plotnine
scipy
statsmodels
pingouin  # For advanced statistical testing
```

## Citation

If using this code or methodology, please cite the original paper:
> Exploring rural women's healthcare access through social vulnerability profiles: A cluster analysis of regional survey data in Texas

## Notes

- The notebooks should be run in sequence starting with `wic_data_engineering.ipynb`
- **For chart generation**: Run `3_factor.ipynb` first to generate cluster data, then `wic_plots.ipynb` for visualizations
- All data files are included in the `/data` directory
- Visualizations are saved in the `/visualizations` directory
- The analysis was conducted on 159 survey responses from rural Texas women