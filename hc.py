# hierarchical_clustering_pipeline.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier


# ======================================================
# 1. Load Data
# ======================================================
def load_data(path):
    dataset = pd.read_csv(path, sep=',', encoding='ISO-8859-1')
    print(dataset.info())
    return dataset


# ======================================================
# 2. Data Preprocessing
# ======================================================
def preprocess_data(df):
    cols_with_invalid_zeros = ["Glucose", "BloodPressure", "BMI"]
    df_treated = df.copy()
    df_treated[cols_with_invalid_zeros] = df_treated[cols_with_invalid_zeros].replace(0, np.nan)

    # mean impute
    mean_cols = ["Glucose", "BMI", "BloodPressure"]
    mean_imp = SimpleImputer(strategy='mean')
    df_treated[mean_cols] = mean_imp.fit_transform(df_treated[mean_cols])

    # drop Insulin & SkinThickness
    df_treated = df_treated.drop(columns=['Insulin', 'SkinThickness'])

    # Feature engineering
    df_treated['Glucose_BMI_Ratio'] = df_treated['Glucose'] / (df_treated['BMI'] + 1e-6)
    df_treated['Age_DPF'] = df_treated['Age'] * df_treated['DiabetesPedigreeFunction']
    df_treated['BP_Age_Ratio'] = df_treated['BloodPressure'] / (df_treated['Age'] + 1)
    df_treated['Glucose_Age'] = df_treated['Glucose'] * df_treated['Age']
    df_treated['BMI_Age'] = df_treated['BMI'] * df_treated['Age']

    return df_treated


# ======================================================
# 3. Feature Selection (Gradient Boosting)
# ======================================================
def feature_selection(df_treated):
    features_all = [
        "Glucose", "BloodPressure", "BMI",
        "DiabetesPedigreeFunction", "Age",
        "Glucose_BMI_Ratio", "Age_DPF", "BP_Age_Ratio",
        "Glucose_Age", "BMI_Age", "Pregnancies"
    ]
    X_fs = df_treated[features_all]
    y_fs = df_treated['Outcome']

    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_fs, y_fs)

    importances = pd.Series(gb_model.feature_importances_, index=features_all).sort_values(ascending=False)
    print("=== Gradient Boosting Feature Importances ===")
    print(importances)

    top_features = importances.head(6).index.tolist()
    print("\nTop Features for Clustering (Gradient Boosting):", top_features)

    return top_features


# ======================================================
# 4. Agglomerative Clustering
# ======================================================
def try_agglomerative(X_scaled, y_true, ks=[2, 3, 4, 5]):
    results = {}
    for k in ks:
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X_scaled) + 1

        sil = silhouette_score(X_scaled, labels)

        # accuracy via majority vote
        cluster_to_label = {}
        for cl in np.unique(labels):
            mask = labels == cl
            maj_label = np.bincount(y_true[mask]).argmax()
            cluster_to_label[cl] = maj_label
        y_pred = np.array([cluster_to_label[cl] for cl in labels])
        accuracy = accuracy_score(y_true, y_pred)

        results[k] = {
            'model': model,
            'labels': labels,
            'silhouette': sil,
            'accuracy': accuracy,
            'cluster_counts': np.bincount(labels)
        }

        print(f'k={k:2d}  silhouette={sil:.4f}  accuracy={accuracy:.4f}  cluster_counts={np.bincount(labels)}')
    return results


# ======================================================
# 5. Interpretasi Clusters
# ======================================================
def interpret_clusters(df_treated, labels, top_features):
    df_clust = df_treated.copy().reset_index(drop=True)
    df_clust['cluster'] = labels
    df_clust['Outcome'] = df_treated['Outcome'].reset_index(drop=True)

    summary_cols = [c for c in top_features if c in df_clust.columns]
    cluster_summary = df_clust.groupby('cluster')[summary_cols].agg(['mean']).round(2)
    print("=== Cluster Means (selected features) ===")
    print(cluster_summary)
    print()

    prevalence = df_clust.groupby('cluster')['Outcome'].mean().rename('diabetic_prevalence')
    counts = df_clust.groupby('cluster').size().rename('count')
    cluster_meta = pd.concat([counts, prevalence], axis=1)
    cluster_meta['diabetic_prevalence_pct'] = (cluster_meta['diabetic_prevalence'] * 100).round(2)
    print("=== Cluster Meta (size & diabetic prevalence) ===")
    print(cluster_meta)
    print()

    overall_means = df_clust[summary_cols].mean()
    interpretations = {}

    def risk_label_from_prev(pct):
        if pct >= 70:
            return "Very High"
        elif pct >= 50:
            return "High"
        elif pct >= 25:
            return "Medium"
        else:
            return "Low"

    for cl in sorted(df_clust['cluster'].unique()):
        desc = []
        cl_means = df_clust[df_clust['cluster'] == cl][summary_cols].mean()
        for feat in summary_cols:
            diff = cl_means[feat] - overall_means[feat]
            pct = diff / overall_means[feat] if overall_means[feat] != 0 else 0
            if pct >= 0.40:
                desc.append(f"{feat} sangat tinggi ({cl_means[feat]:.1f})")
            elif pct >= 0.15:
                desc.append(f"{feat} tinggi ({cl_means[feat]:.1f})")
            elif pct <= -0.40:
                desc.append(f"{feat} sangat rendah ({cl_means[feat]:.1f})")
            elif pct <= -0.15:
                desc.append(f"{feat} rendah ({cl_means[feat]:.1f})")

        prev_pct = cluster_meta.loc[cl, 'diabetic_prevalence_pct']
        risk_label = risk_label_from_prev(prev_pct)

        short = f"Cluster {cl} (n={int(cluster_meta.loc[cl, 'count'])}): prevalensi diabetes {prev_pct}%. Risk level: {risk_label}."
        if desc:
            short += " Ciri utama: " + "; ".join(desc) + "."
        interpretations[cl] = short

    print("=== Interpretasi per Cluster ===")
    for cl, txt in interpretations.items():
        print(txt)

    return df_clust


# ======================================================
# 6. Main Pipeline
# ======================================================
def main():
    df = load_data("data/data_science/pima_indians_diabetes_with_header.csv")
    df_treated = preprocess_data(df)

    top_features = feature_selection(df_treated)

    X = df_treated[top_features]
    y_true = df_treated['Outcome'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = try_agglomerative(X_scaled, y_true, ks=[2, 3, 4, 5])
    best_k = max(results.items(), key=lambda kv: kv[1]['accuracy'])[0]
    labels = results[best_k]['labels']

    df_clust = interpret_clusters(df_treated, labels, top_features)

    # Visualization examples (optional)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    for cl in sorted(np.unique(labels)):
        idx = labels == cl
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Cluster {cl}', alpha=0.6, s=30)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA projection of clusters (k={best_k})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
