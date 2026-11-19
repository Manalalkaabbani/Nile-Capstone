"""
STEP 6  -  PART 5: Find the hidden customer segments (Clustering + PCA).  File: usage_behavior.csv (no labels)

Run:  python 06_clustering_pca.py          (needs step 01)
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import CLEAN_DIR, RANDOM_STATE, header, save_fig, show

# ---------------------------------------------------------------------------
header("5.1  Prepare: drop customer_id, (missing values were handled in step 01), scale")
usage = pd.read_csv(CLEAN_DIR / "usage_clean.csv")
features = usage.drop(columns="customer_id")            # 12 behaviour columns
Z = StandardScaler().fit_transform(features)
print("missing values left:", int(features.isna().sum().sum()), "| scaled shape:", Z.shape)
print("Why scale? KMeans, HAC and DBSCAN work with DISTANCES. Without scaling, avg_daily_data_mb (up to thousands)")
print("would drown topups_per_month (0 to 8) and we would just get 'data-volume groups'.")

# ---------------------------------------------------------------------------
header("5.2  KMeans: k = 2..8, elbow curve and silhouette score")
ks = range(2, 9)
inertia, silhouette = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(Z, km.labels_))
K = 5     # chosen k: see the table and the plots (justify it in your report)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(list(ks), inertia, marker="o"); axes[0].axvline(K, color="crimson", ls="--", lw=1)
axes[0].set(title="Elbow curve (KMeans)", xlabel="Number of clusters k", ylabel="Inertia")
axes[1].plot(list(ks), silhouette, marker="o", color="seagreen"); axes[1].axvline(K, color="crimson", ls="--", lw=1)
axes[1].set(title="Silhouette score (KMeans)", xlabel="Number of clusters k", ylabel="Silhouette")
plt.tight_layout()
save_fig("5_2_elbow_silhouette")

summary = pd.DataFrame({"inertia": inertia, "silhouette": silhouette,
                        "inertia drop vs previous k (%)": [np.nan] + [100 * (inertia[i - 1] - inertia[i]) / inertia[i - 1] for i in range(1, len(inertia))],
                        "cluster sizes": [sorted(np.bincount(KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)).tolist(), reverse=True) for k in ks]},
                       index=list(ks))
show(summary.round(3), "k = 2..8", save_as="06_kmeans_k")
print(f"\nChosen k = {K}: silhouette is near its maximum, the elbow flattens after 5, and k=5 gives four big clean groups plus one tiny extreme group.")

km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Z)
usage["cluster"] = km.labels_
profile = features.groupby(usage["cluster"]).mean()
profile.insert(0, "customers", usage["cluster"].value_counts().sort_index())
profile.insert(1, "share %", (profile["customers"] / len(usage) * 100).round(1))
show(profile.round(1), "cluster profile = mean of each feature", save_as="06_cluster_profile")

profile_z = (features.groupby(usage["cluster"]).mean() - features.mean()) / features.std()
fig, ax = plt.subplots(figsize=(11, 3.8))
sns.heatmap(profile_z, annot=features.groupby(usage["cluster"]).mean().round(1), fmt="", cmap="coolwarm", center=0,
            ax=ax, annot_kws={"size": 8}, cbar_kws={"label": "SD from the average customer"})
ax.set(title="KMeans (k=5) cluster profiles: colour = above/below average, number = mean value", xlabel="", ylabel="Cluster")
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
save_fig("5_2_cluster_profiles")

# ---------------------------------------------------------------------------
header("5.3  Hierarchical clustering (HAC): dendrogram + other linkages")
Zw = linkage(Z, method="ward")
fig, ax = plt.subplots(figsize=(11, 4.5))
dendrogram(Zw, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, show_contracted=True, ax=ax,
           color_threshold=Zw[-(K - 1), 2])
ax.axhline(Zw[-(K - 1), 2] - 1e-9, color="crimson", ls="--", lw=1)
ax.set(title="Dendrogram (Ward linkage, last 30 merges). Red line = cut into 5 clusters",
       xlabel="Cluster (number of customers in brackets)", ylabel="Merge distance")
plt.tight_layout()
save_fig("5_3_dendrogram")

hac_labels, rows = {}, []
for how in ["ward", "average", "complete"]:
    labels = AgglomerativeClustering(n_clusters=K, linkage=how).fit_predict(Z)
    hac_labels[how] = labels
    rows.append({"linkage": how, "cluster sizes": sorted(np.bincount(labels).tolist(), reverse=True),
                 "silhouette": silhouette_score(Z, labels),
                 "Adjusted Rand vs KMeans (1 = same groups)": adjusted_rand_score(usage["cluster"], labels)})
show(pd.DataFrame(rows).set_index("linkage").round(3), "HAC vs KMeans", save_as="06_hac_vs_kmeans")
print("\nWard vs KMeans (rows = KMeans cluster, columns = Ward cluster):")
print(pd.crosstab(usage["cluster"], hac_labels["ward"]))

# ---------------------------------------------------------------------------
header("5.4  DBSCAN: eps from a k-distance plot")
MIN_SAMPLES = 13       # number of features + 1 (a core point needs more neighbours than dimensions)
EPS = 1.6              # the knee of the k-distance plot
kdist = np.sort(NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Z).kneighbors(Z)[0][:, -1])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(kdist); ax.axhline(EPS, color="crimson", ls="--", label=f"chosen eps = {EPS}")
ax.set(title=f"k-distance plot (k = {MIN_SAMPLES})", xlabel="Customers, sorted by distance",
       ylabel=f"Distance to the {MIN_SAMPLES}th nearest neighbour")
ax.legend()
save_fig("5_4_k_distance")

rows = []
for eps in [1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5]:
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(Z)
    rows.append({"eps": eps, "clusters": len(set(lab)) - (1 if -1 in lab else 0), "noise points": int((lab == -1).sum()),
                 "cluster sizes": sorted(np.bincount(lab[lab >= 0]).tolist(), reverse=True) if (lab >= 0).any() else []})
show(pd.DataFrame(rows).set_index("eps"), "what a wrong eps does (too small = too much noise, too big = one giant cluster)")

db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(Z)
usage["dbscan"] = db.labels_
print(f"\nDBSCAN (eps={EPS}, min_samples={MIN_SAMPLES}): {len(set(db.labels_)) - 1} clusters, "
      f"{(db.labels_ == -1).sum()} noise points ({(db.labels_ == -1).mean():.1%})")

noise = usage["dbscan"] == -1
who = pd.DataFrame({f"Noise points (n={noise.sum()})": features[noise].mean(), "Everyone else": features[~noise].mean()})
who["Ratio"] = who.iloc[:, 0] / who.iloc[:, 1]
show(who.round(2), "who are the noise points?", save_as="06_dbscan_noise")
print("KMeans cluster of the noise points:", usage.loc[noise, "cluster"].value_counts().to_dict())

# ---------------------------------------------------------------------------
header("5.5  PCA: explained variance and how many components keep ~90%")
pca = PCA(random_state=RANDOM_STATE).fit(Z)
cumulative = np.cumsum(pca.explained_variance_ratio_)
n90 = int(np.argmax(cumulative >= 0.90) + 1)
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(range(1, len(cumulative) + 1), cumulative * 100, marker="o"); ax.axhline(90, color="crimson", ls="--", lw=1)
ax.set(title="PCA: cumulative explained variance", xlabel="Number of components", ylabel="Variance kept (%)",
       xticks=range(1, len(cumulative) + 1))
save_fig("5_5_pca_variance")
print(f"{n90} components keep {cumulative[n90 - 1]:.1%} of the information (4 components keep {cumulative[3]:.1%}).")
loadings = pd.DataFrame(pca.components_[:3].T, index=features.columns, columns=["PC1", "PC2", "PC3"]).round(2)
show(loadings.loc[loadings.abs().max(axis=1).sort_values(ascending=False).index].head(8), "what PC1, PC2, PC3 mean")

# ---------------------------------------------------------------------------
header("5.7a  Name the segments (rules on the cluster profile, so names stay right if the cluster numbers change)")
means = features.groupby(usage["cluster"]).mean()
sizes = usage["cluster"].value_counts()
names, left = {}, set(means.index)
names[sizes.idxmin()] = "Night-Heavy Outliers"; left.discard(sizes.idxmin())
for column, segment in [("roaming_days_year", "Global Talkers"),
                        ("streaming_hours_week", "Night Gamers & Streamers"),
                        ("social_media_hours_week", "Social Scrollers")]:
    best = means.loc[list(left), column].idxmax()
    names[best] = segment; left.discard(best)
names[left.pop()] = "Light Top-uppers"
usage["segment"] = usage["cluster"].map(names)

seg_table = features.groupby(usage["segment"]).mean()
seg_table.insert(0, "customers", usage["segment"].value_counts())
seg_table.insert(1, "share %", (seg_table["customers"] / len(usage) * 100).round(1))
show(seg_table.sort_values("customers", ascending=False).round(1), "segment profiles", save_as="06_segment_profiles")

# 2D PCA scatter coloured by the best clustering
P2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Z)
colors = dict(zip(sorted(usage["segment"].unique()), sns.color_palette("deep", K)))
fig, ax = plt.subplots(figsize=(9, 6.5))
for seg, g in usage.groupby("segment"):
    outlier = "Outliers" in seg
    ax.scatter(P2[g.index, 0], P2[g.index, 1], s=45 if outlier else 14, alpha=1 if outlier else 0.6,
               label=f"{seg} ({len(g)})", color=colors[seg], edgecolor="k" if outlier else "none")
ax.set(title="Customer segments in PCA space (KMeans, k=5)",
       xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
ax.legend(markerscale=1.5, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
save_fig("5_5_pca_scatter")

# ---------------------------------------------------------------------------
header("5.6  KMeans again on the PCA output")
Zp = PCA(n_components=n90, random_state=RANDOM_STATE).fit_transform(Z)
t0 = time.perf_counter(); km_full = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Z); t_full = time.perf_counter() - t0
t0 = time.perf_counter(); km_pca = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Zp); t_pca = time.perf_counter() - t0
print(f"Adjusted Rand Index (12 features vs {n90} PCA components): {adjusted_rand_score(km_full.labels_, km_pca.labels_):.3f}  (1.0 = identical)")
print(f"Silhouette: {silhouette_score(Z, km_full.labels_):.3f} (12 features) vs {silhouette_score(Zp, km_pca.labels_):.3f} (PCA space; different space, not a fair comparison)")
print(f"Dimensions 12 -> {n90} | variance kept {cumulative[n90 - 1]:.1%} | fit time {t_full:.2f}s vs {t_pca:.2f}s")
print(pd.crosstab(usage["segment"], km_pca.labels_, rownames=["segment (12 features)"], colnames=["PCA-KMeans cluster"]))
print("Gained: fewer columns, less noise, easy plots.  Lost: some variance, and components are blends of columns (harder to read).")

# ---------------------------------------------------------------------------
header("5.7b  Segment names and offers (write your own two lines per segment)")
print("""
  Night Gamers & Streamers : night data + gaming / streaming bundle; home internet or router upgrade.
  Social Scrollers         : social-media pass with a small data pack.
  Light Top-uppers         : small daily / weekly mini-packs, top-up bonus; avoid expensive bundles.
  Global Talkers           : Roaming + International Calling 'Traveller pack'; video-call data pack.
  Night-Heavy Outliers     : no marketing offer; review for resale / fair use (offer a business plan if legitimate).""")

usage[["customer_id", "cluster", "segment", "dbscan"]].to_csv(CLEAN_DIR / "usage_segments.csv", index=False)
print(f"\nSaved the segment of each customer: {CLEAN_DIR / 'usage_segments.csv'}")
print("Next: python 07_association_rules.py")
