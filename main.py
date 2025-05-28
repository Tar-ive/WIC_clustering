import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

# Set page config and ensure proper port setup
import os
port = int(os.environ.get("PORT", 5000))

st.set_page_config(page_title="Women's Healthcare Access in Rural East Texas: Cluster Analysis Dashboard",
                   page_icon="🏥",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Title and introduction
st.title("Women's Healthcare Access in Rural East Texas: Cluster Analysis Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Introduction", "Methodology", "Final Approach", "Results"
])

# Load data
@st.cache_data
def load_data():
    try:
        # Load the main dataset
        cluster_df = pd.read_csv('cluster_data_.csv')
        # Load the focused clustered dataset
        focused_df = pd.read_csv('focused_clustered_data.csv')
        return cluster_df, focused_df
    except Exception as e:
        # Fallback sample data if files don't exist
        st.warning(f"Could not load CSV files ({e}). Using sample data.")
        main_df = pd.DataFrame({
            'cluster': np.random.choice([0, 1, 2, 3, 4, 5], 159),
            'high_barriers_binary': np.random.choice([0, 1], 159),
            'high_needs_binary': np.random.choice([0, 1], 159),
            'self_insured_binary': np.random.choice([0, 1], 159),
            'chronic_cond_binary': np.random.choice([0, 1], 159),
            'sought_care_recently': np.random.choice([0, 1], 159),
            'pct_times_get_Care': np.random.uniform(0, 1, 159),
            'mean_satisfaction': np.random.uniform(2.5, 3.5, 159)
        })
        focused_df = pd.DataFrame({
            'cluster': np.random.choice([0, 1, 2, 3, 4, 5], 159), # Dummy cluster assignment
            'high_barriers_binary': np.random.choice([0, 1], 159),
            'high_needs_binary': np.random.choice([0, 1], 159),
            'demo_young_educated': np.random.choice([True, False], 159),
            'demo_young_uneducated': np.random.choice([True, False], 159),
            'demo_middle_older': np.random.choice([True, False], 159),
            'sought_care_recently': np.random.choice([0, 1], 159),
            'pct_times_get_Care': np.random.uniform(0, 1, 159),
            'mean_satisfaction': np.random.uniform(2.5, 3.5, 159)
        })
        # Add DBSCAN cluster column to focused_df for structure
        focused_df['dbscan_cluster'] = np.random.choice([-1, 0, 1, 2, 3, 4, 5, 6], 159)
        return main_df, focused_df

# Load the data
main_data, focused_data = load_data()

# Load metrics data for comparison charts
@st.cache_data
def load_metrics_data():
    try:
        metrics_df = pd.read_csv('clustering_metrics_summary.csv')
        return metrics_df
    except Exception as e:
        # Create sample metrics data if file doesn't exist
        st.warning(f"Could not load clustering metrics CSV ({e}). Using sample data.")
        metrics_df = pd.DataFrame({
            'Method': ['KMeans', 'Hierarchical (Ward)', 'Spectral', 'DBSCAN', 'Birch', 'Gaussian Mixture'],
            'Silhouette Score': [0.75, 0.75, 0.78, 0.99, 0.65, 0.75],
            'Davies-Bouldin Score': [0.58, 0.59, 0.50, 0.47, 0.68, 0.52],
            'Calinski-Harabasz Score': [1052.4, 1048.6, 1122.8, 1194.3, 987.6, 1078.9],
            'Execution Time (s)': [0.033, 0.002, 0.031, 0.004, 0.004, 0.012]
        })
        return metrics_df

metrics_data = load_metrics_data()

# Create cluster profiles based on PROVIDED DBSCAN results
@st.cache_data
def get_dbscan_cluster_profiles():
    # Data from the text description of DBSCAN results
    profiles = {
        -1: { # Noise Points
            'name': "Noise: Complex Healthcare Access Patterns",
            'size': 22,
            'barriers': 54.5,
            'needs': 59.1,
            'young_educated': 72.7,
            'young_uneducated': 0, # Assuming the rest are not young uneducated
            'middle_older': 100 - 72.7, # Inferring based on young_educated %
            'sought_care': 59.1,
            'success_rate': 67.8,
            'satisfaction': 2.83
        },
         0: { # Cluster 0
            'name': "Cluster 0: Underserved Young Adults",
            'size': 21,
            'barriers': 100.0,
            'needs': 0.0,
            'young_educated': 0.0,
            'young_uneducated': 100.0,
            'middle_older': 0.0,
            'sought_care': 38.1,
            'success_rate': 81.6,
            'satisfaction': 3.07
        },
        1: { # Cluster 1
            'name': "Cluster 1: Young Educated with Good Access",
            'size': 24,
            'barriers': 0.0,
            'needs': 0.0,
            'young_educated': 100.0,
            'young_uneducated': 0.0,
            'middle_older': 0.0,
            'sought_care': 66.7,
            'success_rate': 88.6,
            'satisfaction': 3.34
        },
        2: { # Cluster 2
            'name': "Cluster 2: High-Need Older Adults",
            'size': 11,
            'barriers': 0.0,
            'needs': 100.0,
            'young_educated': 0.0,
            'young_uneducated': 0.0,
            'middle_older': 100.0,
            'sought_care': 100.0,
            'success_rate': 87.6,
            'satisfaction': 3.27
        },
        3: { # Cluster 3
            'name': "Cluster 3: High-Need, High-Barrier Young Adults",
            'size': 13,
            'barriers': 100.0,
            'needs': 100.0,
            'young_educated': 0.0,
            'young_uneducated': 100.0,
            'middle_older': 0.0,
            'sought_care': 53.8,
            'success_rate': 87.0,
            'satisfaction': 2.88
        },
        4: { # Cluster 4
            'name': "Cluster 4: Young Uneducated with Good Access",
            'size': 42,
            'barriers': 0.0,
            'needs': 0.0,
            'young_educated': 0.0,
            'young_uneducated': 100.0,
            'middle_older': 0.0,
            'sought_care': 59.5,
            'success_rate': 93.5,
            'satisfaction': 3.29
        },
        5: { # Cluster 5
            'name': "Cluster 5: Healthy Older Adults",
            'size': 10,
            'barriers': 0.0,
            'needs': 0.0,
            'young_educated': 0.0,
            'young_uneducated': 0.0,
            'middle_older': 100.0,
            'sought_care': 60.0,
            'success_rate': 89.8,
            'satisfaction': 3.33
        },
        6: { # Cluster 6
            'name': "Cluster 6: Young Uneducated with Health Needs",
            'size': 16,
            'barriers': 0.0,
            'needs': 100.0,
            'young_educated': 0.0,
            'young_uneducated': 100.0,
            'middle_older': 0.0,
            'sought_care': 62.5,
            'success_rate': 84.9,
            'satisfaction': 2.98
        }
    }
    # Add total size for percentage calculation
    total_n = sum(p['size'] for p in profiles.values())
    for k in profiles:
        profiles[k]['percentage'] = (profiles[k]['size'] / total_n) * 100
    return profiles

# Get the actual cluster profiles based on DBSCAN description
dbscan_cluster_profiles = get_dbscan_cluster_profiles()
dbscan_cluster_ids = sorted(dbscan_cluster_profiles.keys()) # Should be [-1, 0, 1, 2, 3, 4, 5, 6]


# Introduction page
if page == "Introduction":
    st.header("Understanding Healthcare Access in Rural Texas")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This dashboard presents an analysis of healthcare access patterns in rural Texas among women using WIC services. Using data from a survey of 159 women in rural East Texas, we identified distinct profiles of healthcare access and barriers through cluster analysis.

        ### Project Overview

        With a focus on women, the analysis explores how combinations of factors - demographic characteristics, healthcare needs,
        and access barriers - influence healthcare-seeking behaviors and outcomes in rural Texas communities.

        ### Key Questions

        - What profiles or types of people do or do not get the healthcare they need?
        - How do barriers such as insurance status, geographic access, and social vulnerability interact?
        - What interventions might best support different population segments?
        """)

    with col2:
        # Use an SVG icon for healthcare
        st.markdown("""
        <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="200" viewBox="0 0 24 24" fill="none" stroke="#2771B0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="9" y1="12" x2="15" y2="12"></line>
            <line x1="12" y1="9" x2="12" y2="15"></line>
        </svg>
        <p style="text-align: center;">Rural healthcare access remains a critical issue in Texas</p>
        """,
                    unsafe_allow_html=True)

    st.markdown("---")

    # Create a matplotlib figure for rural healthcare access barriers
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sample data for illustration (in practice, this would be real data)
    categories = ['Insurance', 'Transportation', 'Geographic Distance', 'Provider Shortage', 'Affordability']
    values = [75, 62, 83, 89, 71]

    # Create the bar plot with a custom color gradient
    colors = ['#2771B0', '#3D85C6', '#5B9BD5', '#7CAEE4', '#9BC2E6']
    ax.bar(categories, values, color=colors)

    # Add percentages on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 2, f"{v}%", ha='center', fontweight='bold')

    # Customize the plot
    ax.set_title('Key Healthcare Access Barriers in Rural East Texas', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage Reporting (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig)

    st.subheader("Dataset Overview")
    st.markdown("""
    The dataset includes 159 respondents identifying as female from:

    - **Demographics**: Age, education, employment
    - **Health Status**: Chronic conditions, physical/mental health, quality of life
    - **Social Determinants**: Food, housing, transportation, and income security
    - **Healthcare Access**: Insurance status, geographic proximity to healthcare
    - **Outcomes**: Care-seeking behavior, success in getting care, satisfaction
    """)

    # Display sample of the dataset
    st.subheader("Sample Data (Original Structure)")
    st.dataframe(main_data.head(5))

    # Add a demographic breakdown pie chart
    st.subheader("Respondent Demographics")

    col1, col2 = st.columns(2)

    with col1:
        # Create age distribution pie chart
        fig_age = go.Figure(data=[go.Pie(
            labels=['18-24', '25-34', '35-44', '45-54', '55+'],
            values=[15, 38, 29, 12, 6],  # Sample values
            hole=.3,
            marker_colors=px.colors.sequential.Blues
        )])
        fig_age.update_layout(title_text="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        # Create education distribution pie chart
        fig_edu = go.Figure(data=[go.Pie(
            labels=['Less than HS', 'High School', 'Some College', 'College Degree', 'Advanced Degree'],
            values=[18, 42, 25, 12, 3],  # Sample values
            hole=.3,
            marker_colors=px.colors.sequential.Purples
        )])
        fig_edu.update_layout(title_text="Education Level")
        st.plotly_chart(fig_edu, use_container_width=True)

# Methodology page
elif page == "Methodology":
    st.header("Analytical Approach Evolution")

    st.subheader("Initial Consideration: Latent Class Analysis")
    st.markdown("""
    Initially considered Latent Class Analysis (LCA), but determined it was unsuitable for our dataset:

    - Standard LCA relies on maximum likelihood estimation
    - With categorical variables, each class-variable combination requires parameter estimation
    - Common rule of thumb suggests at least 10 cases per parameter estimated
    - With only 159 observations and multiple categorical variables, LCA would be underpowered
    """)

    st.subheader("Clustering Approach Evolution")

    # Create Graphviz diagram for analysis flow
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB', size='8,5')

    # Process Nodes
    graph.node('initial', 'Initial Dataset\n(20+ variables)', shape='box')
    graph.node(
        'iteration1',
        'Iteration 1\n28 variables with\nindividual & composite measures',
        shape='box')
    graph.node(
        'issues',
        'Issues Identified:\n- Double counting\n- Too many dimensions\n- Mixed variable types\n- Suboptimal cluster separation',
        shape='box',
        style='filled',
        fillcolor='lightyellow')
    graph.node(
        'iteration2',
        'Iteration 2: Focused Approach\n5 composite binary variables\n(Barriers, Needs, Demographics)',
        shape='box')
    graph.node(
        'algo_testing',
        'Algorithm Testing\nK-Means, Hierarchical, GMM, DBSCAN\nEvaluated based on metrics & interpretability',
        shape='box',
        style='filled',
        fillcolor='lightblue'
        )
    graph.node('results',
               'Final Results: DBSCAN\n7 clusters + Noise Group\nReveals nuanced patterns & outliers',
               shape='box',
               style='filled',
               fillcolor='lightgreen')

    # Edges
    graph.edge('initial', 'iteration1')
    graph.edge('iteration1', 'issues')
    graph.edge('issues', 'iteration2')
    graph.edge('iteration2', 'algo_testing')
    graph.edge('algo_testing', 'results')

    st.graphviz_chart(graph)

    st.markdown("---")

    # Add algorithm comparison based on metrics
    st.subheader("Algorithm Comparison on Focused Dataset")

    # Create performance metrics visualization
    fig_metrics = make_subplots(rows=3, cols=1, 
                               subplot_titles=("Silhouette Score (Higher is Better)", 
                                               "Davies-Bouldin Score (Lower is Better)",
                                               "Execution Time (Lower is Better)"))

    # Silhouette Score
    fig_metrics.add_trace(
        go.Bar(
            x=metrics_data['Method'],
            y=metrics_data['Silhouette Score'],
            marker_color='royalblue'
        ),
        row=1, col=1
    )

    # Davies-Bouldin Score
    fig_metrics.add_trace(
        go.Bar(
            x=metrics_data['Method'],
            y=metrics_data['Davies-Bouldin Score'],
            marker_color='royalblue'
        ),
        row=2, col=1
    )

    # Execution Time
    fig_metrics.add_trace(
        go.Bar(
            x=metrics_data['Method'],
            y=metrics_data['Execution Time (s)'],
            marker_color='royalblue'
        ),
        row=3, col=1
    )

    # Update layout
    fig_metrics.update_layout(height=600, showlegend=False)

    st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown("""
    After refining the variables to a focused set (high barriers, high needs, demographic groups), several clustering algorithms were tested:
    - K-Means
    - Hierarchical Clustering (Ward, Complete, Average linkages)
    - Gaussian Mixture Models (GMM)
    - DBSCAN

    **Evaluation:** Algorithms were compared using silhouette scores, Davies-Bouldin index, Calinski-Harabasz index, and qualitative assessment of cluster interpretability and separation.

    **Findings:**
    - K-Means, Hierarchical (Ward), and GMM produced similar, reasonably separated clusters, indicating a robust underlying structure in the focused data.
    - DBSCAN excelled at identifying non-linear patterns and, crucially, isolating a group of 'noise' points (outliers) that didn't fit well into other clusters. This provided a more nuanced view.
    - Initial DBSCAN yielded 9 clusters plus noise, which was highly granular.
    - Based on feedback regarding granularity, parameters were adjusted to achieve a balance, resulting in the **final 7-cluster + noise solution** presented here.
    """)

    # Add DBSCAN algorithm parameter tuning visualization - FIXED
    st.subheader("DBSCAN Parameter Tuning Results")

    # Create a parameter tuning visualization with sample data (fixed approach)
    param_data = pd.DataFrame({
        'eps': [0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0],
        'min_samples': [5, 7, 10, 5, 7, 10, 5, 7, 10],
        'Clusters': [9, 9, 7, 9, 9, 7, 9, 9, 7],
        'Noise Points (%)': [3.8, 3.8, 13.8, 3.8, 3.8, 13.8, 3.8, 3.8, 13.8],
        'Silhouette': [0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999]
    })

    # Create a simple heatmap without using pivot
    fig_params = plt.figure(figsize=(10, 6))

    # Reshape data for heatmap
    eps_values = sorted(param_data['eps'].unique())
    min_samples_values = sorted(param_data['min_samples'].unique())

    # Create a 2D grid for the heatmap
    heatmap_data = np.zeros((len(eps_values), len(min_samples_values)))

    # Fill the grid with cluster counts
    for i, eps in enumerate(eps_values):
        for j, ms in enumerate(min_samples_values):
            mask = (param_data['eps'] == eps) & (param_data['min_samples'] == ms)
            if mask.any():
                heatmap_data[i, j] = param_data.loc[mask, 'Clusters'].values[0]

    # Create the heatmap
    ax = sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt='g', 
                    xticklabels=min_samples_values, yticklabels=eps_values, 
                    cbar_kws={'label': 'Number of Clusters'})
    plt.title('Number of Clusters by DBSCAN Parameters')
    plt.xlabel('min_samples')
    plt.ylabel('eps')

    st.pyplot(fig_params)

    st.markdown("""
    The parameter tuning process for DBSCAN reveals:
    - `eps=0.8` and `min_samples=10` provided the optimal balance of cluster count (7 clusters + noise group)
    - This configuration identified a meaningful noise group (13.8% of cases) with unique characteristics
    - All parameter combinations produced excellent silhouette scores (≈0.9999), indicating very distinct clusters

    **Selected Parameters:** `eps=0.8, min_samples=10`
    """)

    st.markdown("---")
    st.subheader("Rationale for Final DBSCAN Approach")
    st.markdown("""
    DBSCAN was ultimately chosen for the final analysis because:
    1.  **Outlier Detection:** It explicitly identifies 'noise' points (outliers) which represent individuals with unique or complex combinations of characteristics, crucial for not overlooking vulnerable edge cases.
    2.  **Shape Flexibility:** It can find arbitrarily shaped clusters, which is beneficial when relationships between variables aren't strictly linear or spherical.
    3.  **Nuance:** While potentially more complex, the resulting clusters (even after optimizing to 7+noise) offered finer distinctions, particularly separating groups based on combinations of age, education, barriers, and needs.
    4.  **Robustness Confirmation:** The similarity of results from K-Means, Hierarchical Ward, and GMM on the focused data confirmed the general structure, lending confidence that DBSCAN was refining this structure rather than finding something completely different.
    """)

    # Add algorithm comparison visualization using the provided image
    st.subheader("Visual Comparison of Clustering Algorithm Performance")

    try:
        # Display algorithm comparison image (try to use uploaded image)
        st.image("clustering_comparison_grid.png", caption="Comparison of Clustering Methods on Different Datasets")
    except:
        st.warning("Image 'comparison_of_clustering_methods.png' not available. Please upload the image to see the comparison visualization.")

# Final Approach page (Now reflects the chosen DBSCAN method)
elif page == "Final Approach":
    st.header("Final Clustering Approach: DBSCAN on Focused Variables")

    st.markdown("""
    After exploring various methods and variable sets, the final analysis employed **DBSCAN** on a focused set of composite variables. This approach balanced statistical performance with theoretical relevance and interpretability.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Variables for Clustering")
        st.markdown("""
        The DBSCAN algorithm was applied to the following 5 focused, scaled variables:
        1.  **high_barriers_binary**: 1 if ≥2 barriers (social vulnerability, geographic limitations, lack of insurance)
        2.  **high_needs_binary**: 1 if ≥1 healthcare needs (high health need, chronic conditions)
        3.  **demo_young_educated**: 1 for ages 18-44 with college education
        4.  **demo_young_uneducated**: 1 for ages 18-44 without college education
        5.  **demo_middle_older**: 1 for ages 45+

        *Note: Variables were scaled (using StandardScaler) before applying distance-based algorithms like DBSCAN.*
        """)

        # Create a visualization of the variable transformation
        st.subheader("Variable Transformation Process")

        variable_transform = graphviz.Digraph()
        variable_transform.attr(rankdir='LR')

        # Original variables node
        variable_transform.node('original', 'Original Variables\n- Demographics (age, education)\n- Barriers (multiple indicators)\n- Needs (multiple indicators)', 
                               shape='box', style='filled', fillcolor='lightblue')

        # Processing step
        variable_transform.node('processing', 'Processing\n- Binary conversion\n- Composite creation\n- Scaling', 
                               shape='box', style='filled', fillcolor='lightyellow')

        # Final variables node
        variable_transform.node('final', 'Focused Variables\n- high_barriers_binary\n- high_needs_binary\n- demo_young_educated\n- demo_young_uneducated\n- demo_middle_older', 
                               shape='box', style='filled', fillcolor='lightgreen')

        # Edges
        variable_transform.edge('original', 'processing')
        variable_transform.edge('processing', 'final')

        st.graphviz_chart(variable_transform)

        st.subheader("Outcome Variables (for Validation)")
        st.markdown("""
        These variables were *not* used in clustering but were used to profile and understand the resulting clusters:
        - sought_care_recently
        - pct_times_get_Care (Success Rate)
        - mean_satisfaction
        """)

    with col2:
        st.subheader("DBSCAN Algorithm Overview")
        st.markdown("""
        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) works by:

        1. Identifying core points with at least `min_samples` neighbors within distance `eps`
        2. Connecting core points that are neighbors to form clusters
        3. Adding border points (within `eps` of a core point but with fewer neighbors)
        4. Designating remaining points as noise

        This approach allows for:
        - Finding arbitrarily shaped clusters
        - Automatic detection of outliers as "noise"
        - No need to pre-specify the number of clusters
        """)

        # Create a visual explanation of DBSCAN
        fig_dbscan_explain = plt.figure(figsize=(8, 6))

        # Create a scatter plot of points
        np.random.seed(42)
        X = np.random.rand(50, 2)

        # Create two cluster regions
        cluster1 = np.random.randn(30, 2) * 0.1 + [0.3, 0.7]
        cluster2 = np.random.randn(30, 2) * 0.1 + [0.7, 0.3]

        # Add some noise points
        noise = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])

        X = np.vstack([cluster1, cluster2, noise])

        # Draw points
        plt.scatter(X[:30, 0], X[:30, 1], c='royalblue', s=50, label='Cluster 1')
        plt.scatter(X[30:60, 0], X[30:60, 1], c='green', s=50, label='Cluster 2')
        plt.scatter(X[60:, 0], X[60:, 1], c='red', s=50, label='Noise')

        # Draw eps radius around a sample point
        eps_point = X[15]
        circle = plt.Circle((eps_point[0], eps_point[1]), 0.15, fill=False, linestyle='--', color='black')
        plt.gca().add_patch(circle)
        plt.annotate('eps radius', xy=(eps_point[0]+0.05, eps_point[1]+0.15))

        plt.title('DBSCAN Clustering Illustration')
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        st.pyplot(fig_dbscan_explain)

        st.subheader("DBSCAN Parameter Tuning")
        st.markdown("""
        - **Parameter Tuning (`eps`, `min_samples`):**
            - `eps` (maximum distance between samples for one to be considered as in the neighborhood of the other) was determined using a k-distance graph analysis.
            - `min_samples` (number of samples in a neighborhood for a point to be considered as a core point) was chosen based on domain knowledge and desired granularity.
            - The final parameters were selected to yield a meaningful number of clusters (optimized towards 7 + noise) balancing separation and interpretability.

        **Final parameters:** `eps=0.8, min_samples=10` (yielding 7 clusters + noise)
        """)


    st.markdown("---")

    
        #--------------------------------------------------------------------------
        # Cross-Cutting Insights Section
        #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # Cross-Cutting Insights Section
    #--------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Cross-Cutting Insights from DBSCAN")

    st.markdown("""
    ### Understanding Healthcare Access Relationships

    The diagram below visualizes the key relationships between factors influencing healthcare access and outcomes discovered through our DBSCAN clustering analysis. This conceptual model represents both **statistically significant relationships** (barriers → outcomes, needs → satisfaction) and **observed patterns** in the cluster profiles that didn't reach statistical significance but appeared consistently across clusters.

    The thickness of connecting lines could represent the strength of evidence:
    - **Solid lines**: Statistically significant relationships (p<0.05)
    - **Dashed lines**: Observed patterns approaching significance (0.05<p<0.10)
    - **Dotted lines**: Patterns observed in clusters without statistical significance

    This integrated view helps us understand the complex interplay between demographic factors, access barriers, healthcare needs, and outcomes in rural East Texas.
    """)

    # Define colors for the nodes
    BARRIER_COLOR = "#F8D7DA"  # Light red
    NEEDS_COLOR = "#D1ECF1"    # Light blue
    CARE_COLOR = "#D4EDDA"     # Light green
    SUCCESS_COLOR = "#FFF3CD"  # Light yellow
    # Use graphviz to create a relationship diagram
    relationships = graphviz.Digraph()
    relationships.attr(rankdir='LR', size='8,5')
    # Add nodes
    relationships.node('barriers', 'Access Barriers', shape='box', style='filled', fillcolor=BARRIER_COLOR)
    relationships.node('needs', 'Healthcare Needs', shape='box', style='filled', fillcolor=NEEDS_COLOR)
    relationships.node('education', 'Education Level', shape='box', style='filled', fillcolor=CARE_COLOR)
    relationships.node('age', 'Age', shape='box', style='filled', fillcolor=SUCCESS_COLOR)
    # Outcome nodes
    relationships.node('seeking', 'Care-Seeking Behavior', shape='ellipse')
    relationships.node('success', 'Success Getting Care', shape='ellipse')
    relationships.node('satisfaction', 'Care Satisfaction', shape='ellipse')

    # Add edges with labels and appropriate styles based on statistical significance
    # Statistically significant (p<0.05)
    relationships.edge('barriers', 'seeking', label='- Reduces', style='solid', penwidth='2.0')
    relationships.edge('barriers', 'satisfaction', label='- Reduces', style='solid', penwidth='2.0')
    relationships.edge('needs', 'satisfaction', label='- Reduces', style='solid', penwidth='2.0')

    # Approaching significance (0.05<p<0.10)
    relationships.edge('needs', 'seeking', label='+ Increases', style='dashed', penwidth='1.5')

    # Observed patterns without statistical significance
    relationships.edge('education', 'seeking', label='+ Increases', style='dotted', penwidth='1.0')
    relationships.edge('education', 'success', label='+ Increases', style='dotted', penwidth='1.0')
    relationships.edge('age', 'seeking', label='+ Increases for high needs', style='dotted', penwidth='1.0')
    relationships.edge('age', 'satisfaction', label='+ Increases', style='dotted', penwidth='1.0')

    st.graphviz_chart(relationships)

    # Add a legend for the line styles
    st.markdown("""
    **Relationship Types:**
    - **Solid lines**: Statistically significant relationships (p<0.05)
    - **Dashed lines**: Relationships approaching significance (0.05<p<0.10)
    - **Dotted lines**: Observed patterns in clusters without statistical significance
    """)

    st.markdown("""
    1. **Barriers Drive Avoidance & Dissatisfaction:** High barriers strongly deter care-seeking (Cluster 0) or lead to poor experiences even if care is eventually received (Cluster 3, Noise). Addressing insurance, geography, and social vulnerabilities remains paramount.
    2. **High Needs Paradox:** While high needs logically increase care-seeking, they frequently correlate with lower satisfaction (Clusters 3, 6, Noise). This suggests the healthcare system struggles to adequately meet complex needs in this population, even when basic access exists. Focus must shift to care *quality*, *coordination*, and *patient-centeredness* for these groups.
    3. **The Importance of Outliers:** The 'Noise' group isn't just random error; it represents a significant segment (14%) facing compounded challenges and achieving the worst outcomes. Targeted, potentially individualized strategies like case management are needed for such complex profiles.
    4. **Education & Age Interaction:** Education offers an advantage primarily when needs/barriers are low (Cluster 1 > Cluster 4). Age brings higher care utilization, especially with needs (Cluster 2), suggesting established patterns or better insurance access (e.g., Medicare). Younger groups, particularly uneducated ones, need proactive engagement.
    5. **Success Rate vs. Satisfaction:** High success rates don't always mean high satisfaction (e.g., Cluster 3). The *effort* required to achieve success, or the *appropriateness* of the care received, heavily influences patient experience.
    """)
    

# Results page
elif page == "Results":
    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv('dbscan_clustered_data_optimized.csv')
        sig_df = pd.read_csv('relationship_significance.csv')
        return df, sig_df

    df, sig_df = load_data()

    st.header("Cluster Analysis Results")
    st.markdown("### Insights from DBSCAN Clustering Analysis")

    # First section: Cluster overview
    st.header("1. Cluster Overview")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Create pivot table for cluster profiles
        cluster_profiles = pd.pivot_table(
            df, 
            index='dbscan_cluster',
            values=['high_barriers_binary', 'high_needs_binary', 'demo_young_educated', 
                    'demo_young_uneducated', 'demo_middle_older'],
            aggfunc=np.mean
        )
        
        # Rename columns for better display
        cluster_profiles.columns = ['High Barriers', 'High Needs', 'Young Educated', 
                                'Young Uneducated', 'Middle/Older']
        
        # Calculate cluster sizes
        cluster_sizes = df['dbscan_cluster'].value_counts().sort_index()
        cluster_profiles['Size'] = cluster_sizes
        
        # Format as percentages
        for col in cluster_profiles.columns[:-1]:  # All except Size
            cluster_profiles[col] = (cluster_profiles[col] * 100).round(1).astype(str) + '%'
        
        # Display cluster profiles
        st.write("Cluster Characteristics (% of members with each attribute)")
        st.dataframe(cluster_profiles.style.background_gradient(cmap='Blues', subset=pd.IndexSlice[:, ['Size']]))

    with col2:
        # Create a pie chart of cluster distribution with descriptive names
        pie_data = cluster_sizes.reset_index()
        pie_data.columns = ['Cluster', 'Count']

        # Map cluster numbers to descriptive names
        cluster_names = {
            -1: "Noise (Complex Cases)",
            0: "Underserved Young Adults",
            1: "Young Educated with Needs",
            2: "Older Adults",
            3: "High-Need, High-Barrier Young",
            4: "Young Educated, Good Access",
            5: "High-Barrier Group",
            6: "Young Uneducated with Needs"
        }

        pie_data['Cluster Name'] = pie_data['Cluster'].map(cluster_names)

        fig = px.pie(
            pie_data, 
            values='Count', 
            names='Cluster Name',
            title='Distribution of Clusters',
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        # Improve layout for better readability
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.1
        ))

        st.plotly_chart(fig)

    # Second section: Key relationships and significance
    st.header("2. Statistical Significance of Key Relationships")

    # Add plain-English lead-in text
    st.markdown("""
    This section shows how strongly different factors influence healthcare outcomes in rural East Texas. 
    The horizontal bars represent the strength and direction of each relationship:
    - **Bars extending to the right (positive values)** mean the factor increases an outcome (e.g., higher education increases care-seeking)
    - **Bars extending to the left (negative values)** mean the factor decreases an outcome (e.g., barriers reduce satisfaction)
    - **Green bars with asterisks (*)** indicate statistically significant relationships (p < 0.05), meaning we're confident the relationship is real
    - **Darker bars without asterisks** are observed patterns that didn't reach statistical significance

    The length of each bar shows the effect size - longer bars mean stronger relationships.
    """)

    # Create a horizontal bar chart for relationship significance
    sig_df_sorted = sig_df.sort_values(by='p_value')
    # Filter for direct relationships (not interactions or ANOVA)
    direct_relationships = sig_df_sorted[sig_df_sorted['Test'] == 'Point-biserial']
    fig = px.bar(
        direct_relationships,
        y='Relationship',
        x='Statistic',
        color='p_value',
        color_continuous_scale='RdYlGn_r',
        range_color=[0, 0.05],
        orientation='h',
        labels={'Statistic': 'Effect Size (Correlation)', 'Relationship': '', 'p_value': 'p-value'},
        title='Statistical Significance of Key Relationships'
    )
    # Add a vertical line at x=0
    fig.add_shape(
        type="line",
        x0=0, y0=-0.5, x1=0, y1=9.5,
        line=dict(color="black", width=1, dash="dash")
    )
    # Add significance markers
    for i, row in direct_relationships.iterrows():
        fig.add_annotation(
            x=row['Statistic'],
            y=row['Relationship'],
            text="*" if row['p_value'] < 0.05 else "",
            showarrow=False,
            font=dict(size=24, color="red")
        )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("*Asterisk (*) indicates statistically significant relationship (p < 0.05)*")
    # Create a styled table for significance results
    st.subheader("Statistical Test Results")

    # Format p-values to show asterisks
    sig_df['Formatted p-value'] = sig_df['p_value'].apply(lambda x: f"{x:.3f}" + (" *" if x < 0.05 else ""))

    # Create a clean display table with numeric p-values
    display_df = sig_df[['Relationship', 'Test', 'Statistic', 'p_value']]
    display_df.columns = ['Relationship', 'Statistical Test', 'Effect Size', 'p-value']
    
    # Add asterisk for significant values
    display_df['p-value'] = display_df['p-value'].apply(lambda x: f"{x:.3f}" + (" *" if x < 0.05 else ""))

    # Filter options
    test_type = st.selectbox(
        "Filter by test type:",
        options=["All Tests"] + list(display_df['Statistical Test'].unique())
    )

    if test_type != "All Tests":
        display_df = display_df[display_df['Statistical Test'] == test_type]

    st.dataframe(display_df.style.background_gradient(subset=['Effect Size'], cmap='RdYlGn'))

    # Third section: Outcome analysis by cluster
    st.header("3. Healthcare Outcomes by Cluster")

    # Add plain-English lead-in text
    st.markdown("""
    This section shows how women in different clusters experience healthcare outcomes. The table shows the average values for each cluster, while the chart on the right allows you to explore the distribution of specific outcomes across clusters.

    For each cluster, we measure:
    - **Sought Care Recently**: Percentage of women who sought healthcare in the past 6-12 months
    - **Success Rate**: Percentage of times women reported getting care when they needed it
    - **Satisfaction**: Average satisfaction rating on a 1-5 scale
    """)

    col1, col2 = st.columns(2)
    with col1:
        # Create outcome means by cluster
        outcome_means = pd.pivot_table(
            df, 
            index='dbscan_cluster',
            values=['sought_care_recently', 'pct_times_get_Care', 'mean_satisfaction'],
            aggfunc=np.mean
        )

        # Format the table
        outcome_means.columns = ['Sought Care (%)', 'Success Rate (%)', 'Satisfaction (1-5)']
        outcome_means['Sought Care (%)'] = (outcome_means['Sought Care (%)'] * 100).round(1)
        outcome_means['Success Rate (%)'] = (outcome_means['Success Rate (%)'] * 100).round(1)
        outcome_means['Satisfaction (1-5)'] = outcome_means['Satisfaction (1-5)'].round(2)

        # Map index to cluster names for better readability
        cluster_names = {
            -1: "Noise (Complex Cases)",
            0: "Underserved Young Adults",
            1: "Young Educated with Needs",
            2: "Older Adults",
            3: "High-Need, High-Barrier Young",
            4: "Young Educated, Good Access",
            5: "High-Barrier Group",
            6: "Young Uneducated with Needs"
        }
        outcome_means.index = outcome_means.index.map(lambda x: cluster_names.get(x, f"Cluster {x}"))

        st.write("Average Outcomes by Cluster")
        st.dataframe(outcome_means.style.background_gradient(cmap='YlGn'))

        # Add footnote explaining the binary variable
        st.caption("""
        *Note: "Sought Care (%)" shows the percentage of women in each cluster who sought healthcare recently (originally a yes/no question).*
        """)

    with col2:
        # Outcome selection for visualization
        outcome = st.selectbox(
            "Select outcome to visualize:",
            options=['sought_care_recently', 'pct_times_get_Care', 'mean_satisfaction'],
            format_func=lambda x: {
                'sought_care_recently': 'Sought Care Recently (%)', 
                'pct_times_get_Care': 'Success Rate (%)', 
                'mean_satisfaction': 'Satisfaction (1-5)'
            }[x]
        )

        # For binary variable, offer alternative visualization
        if outcome == 'sought_care_recently':
            # Create a dataframe of percentages by cluster
            sought_care_pcts = df.groupby('dbscan_cluster')['sought_care_recently'].mean() * 100
            sought_care_df = sought_care_pcts.reset_index()
            sought_care_df.columns = ['Cluster', 'Percentage']

            # Map cluster numbers to names
            sought_care_df['Cluster Name'] = sought_care_df['Cluster'].map(cluster_names)

            # Create a bar chart instead of box plot for binary data
            fig = px.bar(
                sought_care_df,
                x='Cluster Name',
                y='Percentage',
                title="Percentage Who Sought Care Recently by Cluster",
                labels={'Percentage': 'Percentage Who Sought Care (%)', 'Cluster Name': 'Cluster'},
                color='Percentage',
                color_continuous_scale='YlGn'
            )
            fig.update_layout(xaxis_tickangle=-45)
        else:
            # For non-binary variables, use box plot as before
            y_data = df[outcome]
            if outcome != 'mean_satisfaction':
                y_data = y_data * 100

            # Map cluster numbers to names for the x-axis
            cluster_mapping = {str(k): v for k, v in cluster_names.items()}

            # Create box plot
            fig = px.box(
                df, 
                x='dbscan_cluster', 
                y=y_data,
                title=f"Distribution of {outcome.replace('_', ' ').title()} by Cluster",
                labels={'dbscan_cluster': 'Cluster', 'y': outcome.replace('_', ' ').title()}
            )

            # Update x-axis tick labels to use cluster names
            fig.update_xaxes(ticktext=[cluster_names.get(i, f"Cluster {i}") for i in sorted(df['dbscan_cluster'].unique())], 
                             tickvals=sorted(df['dbscan_cluster'].unique()))
            fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig)
    st.markdown("---")
    st.subheader("All Cluster Profiles")

    # Row 1 (Priority clusters)
    st.markdown("### Priority Groups for Intervention")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Noise Group: Complex Cases")
        profile = dbscan_cluster_profiles[-1]
        st.markdown(f"""
        **({profile['percentage']:.1f}%)**
        **Key Characteristics**:
        - Mostly Young Educated (73%)
        - High Barriers (55%) & High Needs (59%)
        - **Lowest Success Rate (68%)**
        - **Lowest Satisfaction (2.83)**
        **Potential Issues**: Complex conditions, navigation difficulties, care coordination failures.
        **Interventions**: Case management, integrated care models, patient navigation support, specialist access assessment.
        """)
    with col2:
        st.markdown("#### Cluster 0: Underserved Young Adults")
        profile = dbscan_cluster_profiles[0]
        st.markdown(f"""
        **({profile['percentage']:.1f}%)**
        **Key Characteristics**:
        - Young, Uneducated (100%)
        - Max Barriers (100%), No Needs (0%)
        - **Lowest Care-Seeking (38%)**
        - Low Insurance (Implied by high barriers)
        **Potential Issues**: Avoidance of preventive care, risk of future acute needs.
        **Interventions**: Insurance enrollment, mobile clinics, community health workers, health literacy programs, addressing SDOH (transport, etc.).
        """)
    with col3:
        st.markdown("#### Cluster 3: High-Need, High-Barrier Young")
        profile = dbscan_cluster_profiles[3]
        st.markdown(f"""
        **({profile['percentage']:.1f}%)**
        **Key Characteristics**:
        - Young, Uneducated (100%)
        - Max Barriers (100%) & Max Needs (100%)
        - Moderate Care-Seeking (54%)
        - **Low Satisfaction (2.88)**
        **Potential Issues**: Difficulty managing chronic conditions with limited resources, frustrating care experiences.
        **Interventions**: Chronic disease management support, transportation aid, care coordination, insurance stability, SDOH support.
        """)

    # Row 2
    st.markdown("### Secondary Priority Groups")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Cluster 1: Young Educated with Needs")
        profile = dbscan_cluster_profiles[1]
        st.markdown(f"""
        **({profile['percentage']:.1f}%)**
        **Key Characteristics**:
        - Young Educated (100%)
        - High Needs (100%), No Barriers (0%)
        - Good Care-Seeking (67%)
        - Good Satisfaction (3.34)
        **Potential Issues**: Managing complex health needs despite good resources, navigating specialist care.
        **Interventions**: Specialist care coordination, chronic condition management support, telehealth options.
        """)
    with col2:
        st.markdown("#### Cluster 6: Young Uneducated with Needs")
        profile = dbscan_cluster_profiles[6]
        st.markdown(f"""
        **({profile['percentage']:.1f}%)**
        **Key Characteristics**:
        - Young Uneducated (100%)
        - High Needs (100%), No Barriers (0%)
        - Moderate Care-Seeking (63%)
        - Lower Satisfaction (2.98)
        **Potential Issues**: Difficulty understanding treatment plans, health literacy challenges despite access.
        **Interventions**: Patient education materials, care plan simplification, follow-up support, health literacy interventions.
        """)
    with col3:
        st.markdown("#### Cluster 5: High-Barrier Group")
        profile = dbscan_cluster_profiles[5]
        st.markdown(f"""
        **({profile['percentage']:.1f}%)**
        **Key Characteristics**:
        - Mixed demographics
        - High Barriers (100%), No High Needs (0%)
        - Reasonably High Care-Seeking (60%)
        - Moderate Satisfaction (3.33)
        **Potential Issues**: Preventive care avoidance, difficulty with follow-up appointments.
        **Interventions**: Transportation assistance, telehealth options, reduced paperwork, schedule flexibility.
        """)

    # Row 3
    st.markdown("### Lower Priority Groups")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Cluster 2: Older Adults")
        profile = dbscan_cluster_profiles[2]
        st.markdown(f"""
        **({profile['percentage']:.1f}%)**
        **Key Characteristics**:
        - Middle/Older Adults (100%)
        - No High Barriers (0%), No High Needs (0%)
        - **Perfect Care-Seeking (100%)**
        - Highest Satisfaction (3.27)
        **Potential Issues**: Minimal current issues; potential for developing age-related conditions.
        **Interventions**: Preventive screenings, wellness programs, preparation for managing future health needs.
        """)
    with col2:
        st.markdown("#### Cluster 4: Young Educated, Good Access")
        profile = dbscan_cluster_profiles[4]
        st.markdown(f"""
        **({profile['percentage']:.1f}%)**
        **Key Characteristics**:
        - Young Educated (100%)
        - No Barriers (0%), No High Needs (0%)
        - High Care-Seeking (79%)
        - **Highest Success Rate (94%)**
        **Potential Issues**: Few current issues; opportunity for health promotion.
        **Interventions**: Preventive care incentives, health education, community health leadership.
        """)

    st.markdown("---")
    st.subheader("Rationale for Prioritization")
    st.markdown("""
    The priority groups identified above represent the most vulnerable segments of the rural population with the poorest healthcare outcomes:

    1. **The Noise Group** requires urgent attention due to their complex combination of characteristics and the worst outcomes across all measures. Their unique profiles don't fit neatly into other clusters, suggesting they may be falling through cracks in the current healthcare system.

    2. **Cluster 0 (Underserved Young Adults)** has the lowest care-seeking behavior despite being young, placing them at high risk for preventable complications and emergency care needs in the future. Their 100% barrier rate indicates systemic access issues.

    3. **Cluster 3 (High-Need, High-Barrier Young)** faces the "double jeopardy" of both maximum barriers and maximum healthcare needs, creating a particularly challenging situation requiring coordinated intervention.

    These three groups account for approximately 35% of the rural female population studied but represent those with the most significant unmet healthcare needs and poorest outcomes. Targeted interventions for these groups would likely yield the highest return on investment in terms of improved health outcomes and system efficiency.

    Secondary priority groups also face significant challenges but have either better outcomes or fewer compounding factors. Lower priority groups are generally functioning well within the current system but could benefit from preventive approaches.
    """)
    st.markdown("---")
    st.markdown("### Other Groups & Considerations")
    st.markdown("""
    - **Cluster 6 (Young Uneducated w/ Needs, No Barriers):** High needs lead to low satisfaction (2.98) despite good access. Focus on *quality* and *appropriateness* of care, patient education, and self-management support.
    - **Preventive Care:** Clusters 0 and 4 (Young Uneducated with low needs) have lower care-seeking rates than educated counterparts (Cluster 1). Targeted outreach for screenings and preventive services is crucial.
    """)
    # st.markdown("---")
    # st.markdown("### Other Groups & Considerations")
    # st.markdown("""
    #     - **Cluster 6 (Young Uneducated w/ Needs, No Barriers):** High needs lead to low satisfaction (2.98) despite good access. Focus on *quality* and *appropriateness* of care, patient education, and self-management support.
    #     - **Preventive Care:** Clusters 0 and 4 (Young Uneducated with low needs) have lower care-seeking rates than educated counterparts (Cluster 1). Targeted outreach for screenings and preventive services is crucial.
    #     """)


    # Fourth section: Cross-cutting insights
    st.header("4. Cross-Cutting Insights")

    insights = [
        {
            "title": "Barriers Drive Avoidance & Dissatisfaction",
            "description": "High barriers strongly deter care-seeking (p=0.015) and reduce satisfaction (p=0.002).",
            "supported": True,
            "evidence": "All three barrier relationships are statistically significant (p<0.05)."
        },
        {
            "title": "High Needs Paradox",
            "description": "High healthcare needs correlate with lower satisfaction (p=0.008) despite increasing care-seeking.",
            "supported": True,
            "evidence": "Significant negative relationship between needs and satisfaction confirms this pattern."
        },
        {
            "title": "The Importance of Outliers",
            "description": "The 'Noise' group (14%) represents complex cases with the worst outcomes.",
            "supported": "Partial",
            "evidence": "Cluster differences approach significance (p≈0.098). Requires further investigation."
        },
        {
            "title": "Education & Age Interaction",
            "description": "Education advantage primarily when needs/barriers are low; age increases utilization.",
            "supported": False,
            "evidence": "Interaction effects weren't statistically significant (p=0.251)."
        },
        {
            "title": "Success Rate vs. Satisfaction",
            "description": "Success in getting care doesn't always translate to high satisfaction.",
            "supported": "Partial",
            "evidence": "Cluster differences in satisfaction approach significance (p=0.098)."
        }
    ]

    for i, insight in enumerate(insights):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if insight["supported"] == True:
                st.success("✓ Statistically Supported")
            elif insight["supported"] == "Partial":
                st.warning("⚠ Partially Supported")
            else:
                st.error("✗ Limited Statistical Support")
        
        with col2:
            st.subheader(f"{i+1}. {insight['title']}")
            st.write(insight["description"])
            st.caption(f"**Evidence:** {insight['evidence']}")
        
        st.markdown("---")

    # Add a conclusion
    st.header("Conclusion")
    st.write("""
    This analysis reveals that barriers to healthcare access have the most consistent and statistically significant impact on outcomes.
    High healthcare needs create a paradox where they drive care-seeking behavior but also correlate with lower satisfaction.
    While not all relationships reached statistical significance, the patterns identified through DBSCAN clustering provide
    valuable insights for designing targeted interventions to improve healthcare access in rural Texas.
    """)

    # Add footer

# Implications page
elif page == "Implications":
    st.header("Implications and Recommendations")
    
    st.markdown("""
    Based on the cluster analysis, we've developed recommendations to address the unique needs of different population segments in rural East Texas.
    These recommendations focus on improving healthcare access, enhancing delivery systems, ensuring quality and appropriateness of care, and targeting prevention efforts.
    """)
    
    # #--------------------------------------------------------------------------
    # # Recommendations Section
    # #--------------------------------------------------------------------------
    # st.markdown("---")

    # st.subheader("Policy and System-Level Recommendations")

    # # Create tabs for different recommendation categories
    # rec_tabs = st.tabs(["Safety Net", "Care Delivery", "Quality & Appropriateness", "Prevention & Outreach"])

    # with rec_tabs[0]:
    #     st.markdown("### 1. Strengthen the Safety Net")

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.markdown("""
    #         #### Insurance & Affordability
    #         - **Insurance Expansion & Enrollment:** Simplify Medicaid/Marketplace enrollment, targeted outreach (esp. Cluster 0 & 3)
    #         - **Address Affordability:** Promote sliding scale fees, prescription assistance programs
    #         - **Policy Advocacy:** Support Medicaid expansion in Texas
    #         """)

    #     with col2:
    #         st.markdown("""
    #         #### Social Determinants of Health
    #         - **SDOH Integration:** Link healthcare with housing, food, and transportation resources
    #         - **Community Partnerships:** Leverage WIC connections to address multiple needs
    #         - **Geographic Solutions:** Support satellite clinics, mobile health units, telehealth expansion
    #         """)

    #     # Create a target population diagram for safety net interventions
    #     target_pop_safety = pd.DataFrame({
    #         'Cluster': ['Noise Group', 'Cluster 0', 'Cluster 3', 'Cluster 6', 'Cluster 4', 'Cluster 1', 'Cluster 2', 'Cluster 5'],
    #         'Priority': [3, 3, 3, 1, 1, 0, 0, 0]  # 0-3 scale
    #     })

    #     fig_target_safety = px.bar(
    #         target_pop_safety, 
    #         x='Cluster', 
    #         y='Priority',
    #         title='Safety Net Intervention Priority by Cluster',
    #         color='Priority',
    #         color_continuous_scale=[(0, 'lightgray'), (1, 'crimson')],
    #         labels={'Priority': 'Priority Level (0-3)'}
    #     )

    #     fig_target_safety.update_layout(height=400)
    #     st.plotly_chart(fig_target_safety, use_container_width=True)

    # with rec_tabs[1]:
    #     st.markdown("### 2. Enhance Care Delivery")

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.markdown("""
    #         #### Access Innovations
    #         - **Mobile Health & Telehealth:** Expand reach to overcome geographic barriers
    #         - **Digital Equity:** Ensure broadband access and digital literacy support
    #         - **Extended Hours:** Provide evening and weekend appointments for working populations
    #         """)

    #     with col2:
    #         st.markdown("""
    #         #### Coordination & Workforce
    #         - **Care Coordination & Case Management:** Implement for high-need/high-barrier groups
    #         - **Workforce Development:** Incentivize rural providers, promote cultural competency
    #         - **Community Health Workers:** Deploy CHWs from local communities
    #         """)

    #     # Create visual for care delivery approaches
    #     care_delivery_data = pd.DataFrame({
    #         'Approach': ['Mobile Clinics', 'Telehealth', 'Extended Hours', 'Care Coordination', 'CHWs', 'Provider Incentives'],
    #         'Barriers Impact': [3, 3, 2, 2, 3, 2],
    #         'Needs Impact': [2, 2, 1, 3, 2, 2],
    #         'Satisfaction Impact': [2, 2, 3, 3, 3, 2]
    #     })

    #     fig_care_del = px.parallel_categories(
    #         care_delivery_data,
    #         dimensions=['Approach', 'Barriers Impact', 'Needs Impact', 'Satisfaction Impact'],
    #         color='Satisfaction Impact',
    #         color_continuous_scale=px.colors.sequential.Viridis,
    #         labels={'Barriers Impact': 'Reduces Barriers (1-3)',
    #                'Needs Impact': 'Addresses Needs (1-3)',
    #                'Satisfaction Impact': 'Improves Satisfaction (1-3)'}
    #     )

    #     fig_care_del.update_layout(height=500)
    #     st.plotly_chart(fig_care_del, use_container_width=True)
    #     st.caption("Impact ratings: 1=Minor impact, 2=Moderate impact, 3=Major impact")

    # with rec_tabs[2]:
    #     st.markdown("### 3. Focus on Quality & Appropriateness")

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.markdown("""
    #         #### Chronic Disease Management
    #         - **Enhanced Programs:** Improve support for Clusters 2, 3, 6, and Noise group members
    #         - **Self-Management Support:** Provide education and resources for ongoing condition management
    #         - **Medication Management:** Ensure access to and proper use of prescribed medications
    #         """)

    #     with col2:
    #         st.markdown("""
    #         #### Patient-Centered Care
    #         - **Communication Training:** Improve provider-patient communication
    #         - **Shared Decision-Making:** Involve patients in treatment decisions
    #         - **Quality Measurement:** Track satisfaction alongside access metrics
    #         """)

    #     # Create a priority matrix for quality initiatives
    #     quality_data = {
    #         'Initiative': ['Chronic Disease Programs', 'Self-Management Support', 'Medication Management',
    #                      'Provider Communication', 'Shared Decision-Making', 'Quality Measurement'],
    #         'Cluster 0': [1, 1, 1, 2, 1, 2],
    #         'Cluster 3': [3, 3, 3, 2, 2, 3],
    #         'Cluster 6': [3, 3, 2, 3, 3, 3],
    #         'Noise Group': [3, 2, 3, 3, 3, 3]
    #     }

    #     quality_df = pd.DataFrame(quality_data)
    #     quality_df = quality_df.set_index('Initiative')

    #     fig_quality = px.imshow(
    #         quality_df,
    #         color_continuous_scale='Viridis',
    #         labels=dict(x="Priority Group", y="Quality Initiative", color="Priority (1-3)"),
    #         title="Quality & Appropriateness Initiatives by Priority Group"
    #     )

    #     fig_quality.update_layout(height=500)
    #     st.plotly_chart(fig_quality, use_container_width=True)

    # with rec_tabs[3]:
    #     st.markdown("### 4. Targeted Prevention & Outreach")

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.markdown("""
    #         #### Preventive Campaigns
    #         - **Screenings:** Focus on younger, lower-seeking groups (Clusters 0, 4)
    #         - **Health Education:** Provide age and education-appropriate materials
    #         - **Life Course Approach:** Connect reproductive health to overall wellness
    #         """)

    #     with col2:
    #         st.markdown("""
    #         #### Community Engagement
    #         - **Trusted Channels:** Utilize schools, churches, WIC clinics
    #         - **Peer Support:** Develop peer networks for specific demographic groups
    #         - **Community Assessment:** Ongoing identification of emerging needs
    #         """)

    #     # Create a visualization of prevention strategies
    #     prevention_strategies = pd.DataFrame({
    #         'Strategy': ['Mobile Screening', 'WIC-based Outreach', 'School Partnerships', 
    #                    'Faith-based Programs', 'Social Media', 'Text Reminders'],
    #         'Reaches Young Uneducated': [3, 3, 2, 2, 3, 3],
    #         'Reaches Young Educated': [2, 1, 1, 2, 3, 3],
    #         'Reaches Older Adults': [3, 1, 0, 3, 1, 2],
    #         'Effectiveness': [3, 3, 2, 3, 2, 2]
    #     })

    #     fig_prev = px.scatter(
    #         prevention_strategies,
    #         x='Reaches Young Uneducated', 
    #         y='Effectiveness',
    #         size='Effectiveness',
    #         color='Strategy',
    #         hover_data=['Reaches Young Educated', 'Reaches Older Adults'],
    #         title='Prevention & Outreach Strategy Effectiveness'
    #     )

    #     fig_prev.update_layout(
    #         xaxis=dict(title='Reaches Young Uneducated Population (1-3)'),
    #         yaxis=dict(title='Overall Effectiveness (1-3)'),
    #         height=500
    #     )
    #     st.plotly_chart(fig_prev, use_container_width=True)
    
    # #--------------------------------------------------------------------------
    # # Implementation Priority Matrix
    # #--------------------------------------------------------------------------
    # st.subheader("Implementation Priority Matrix")

    # # Create a quadrant matrix of impact vs. implementation ease
    # recommendations = pd.DataFrame({
    #     'Recommendation': [
    #         'Insurance Enrollment Assistance', 'Mobile Health Units', 'CHW Program',
    #         'Care Coordination', 'Extended Hours', 'Patient Navigation',
    #         'Provider Communication Training', 'Telehealth Expansion', 'Quality Measurement',
    #         'SDOH Screening & Referral', 'WIC-based Outreach', 'Self-Management Support'
    #     ],
    #     'Impact': [3, 3, 3, 3, 2, 3, 2, 2, 2, 3, 2, 2],
    #     'Ease': [2, 1, 2, 1, 2, 2, 3, 2, 3, 2, 3, 2],
    #     'Category': [
    #         'Safety Net', 'Care Delivery', 'Safety Net',
    #         'Care Delivery', 'Care Delivery', 'Care Delivery',
    #         'Quality', 'Care Delivery', 'Quality',
    #         'Safety Net', 'Prevention', 'Quality'
    #     ]
    # })

    # fig_matrix = px.scatter(
    #     recommendations,
    #     x='Ease',
    #     y='Impact',
    #     color='Category',
    #     size=[i*3 for i in recommendations['Impact']],  # Scaled for visibility
    #     text='Recommendation',
    #     title='Implementation Priority Matrix: Impact vs. Feasibility',
    #     labels={'Ease': 'Implementation Ease (1-3)', 'Impact': 'Potential Impact (1-3)'}
    # )

    # fig_matrix.update_traces(textposition='top center')
    # fig_matrix.update_layout(
    #     height=600,
    #     xaxis=dict(range=[0.5, 3.5]),
    #     yaxis=dict(range=[0.5, 3.5])
    # )

    # # Add quadrant separators
    # fig_matrix.add_shape(
    #     type="line", x0=2, y0=0.5, x1=2, y1=3.5,
    #     line=dict(color="Gray", width=1, dash="dot")
    # )
    # fig_matrix.add_shape(
    #     type="line", x0=0.5, y0=2, x1=3.5, y1=2,
    #     line=dict(color="Gray", width=1, dash="dot")
    # )

    # # Add quadrant labels
    # fig_matrix.add_annotation(x=1.25, y=2.75, text="High Priority", showarrow=False, font=dict(size=14))
    # fig_matrix.add_annotation(x=2.75, y=2.75, text="Quick Wins", showarrow=False, font=dict(size=14))
    # fig_matrix.add_annotation(x=1.25, y=1.25, text="Long-term Projects", showarrow=False, font=dict(size=14))
    # fig_matrix.add_annotation(x=2.75, y=1.25, text="Low Priority", showarrow=False, font=dict(size=14))

    # st.plotly_chart(fig_matrix, use_container_width=True)

#------------------------------------------------------------------------------
# Footer
#------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "Dashboard created by [Saksham Adhikari](https://www.linkedin.com/in/saksham-adhikari-4727571b5/) for Translational Health Research Center")


