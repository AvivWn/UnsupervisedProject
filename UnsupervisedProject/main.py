from utils import *

import os
if not os.path.exists("figures"):
    os.makedirs("figures")

# # Loading the dataset

# Loading the wine dataset
red_wine_quality_df = pd.read_csv('dataset/winequality-red.csv', header=0, encoding='utf-8', delimiter= ';')
red_wine_quality_df['type'] = 'red'
white_wine_quality_df = pd.read_csv('dataset/winequality-white.csv', header=0, encoding='utf-8', delimiter= ';')
white_wine_quality_df['type'] = 'white'
wine_quality_df = pd.concat([red_wine_quality_df, white_wine_quality_df]).reset_index(drop=True)
wine_quality_df = pd.DataFrame.sample(wine_quality_df, frac=1).reset_index(drop=True)

wine_quality_df.head()


# # Exploring The Dataset

# ### Visualization

for column in wine_quality_df.columns:
    if column != "type":
        sns.kdeplot(wine_quality_df.loc[wine_quality_df.loc[:, 'type'] == 'white', column], shade=True, color=color_by_cluster('white'), label='white')
        fig = sns.kdeplot(wine_quality_df.loc[wine_quality_df.loc[:, 'type'] == 'red', column], shade=True, color=color_by_cluster('red'), label='red')
        fig.figure.suptitle(column)
        plt.savefig(f'{figures_dir}dist_{column.replace(" ", "_")}.png')
        #plt.show()


palette = {1: color_by_cluster('white'), 0: color_by_cluster('red')}
for index, column in np.ndenumerate(wine_quality_df.columns):
    sns.boxplot('type', column, data=wine_quality_df.replace({'red': 0, 'white':1}), palette=palette)
    plt.savefig(f'{figures_dir}boxplot_{column.replace(" ", "_")}.png')
    #plt.show()


# tmp_df = wine_quality_df.drop(["type"], axis=1)
# all_outliers = np.array([],dtype='int64')
# for feature in tmp_df.keys():
#     Q1 = np.percentile(tmp_df[feature],25)
#     Q3 = np.percentile(tmp_df[feature],75)
#     step = 1.5*(Q3-Q1)
#     outlier_pts = tmp_df[ ~((tmp_df[feature]>=Q1-step) & (tmp_df[feature]<=Q3+step))]
#     all_outliers=np.append(all_outliers,outlier_pts.index.values.astype('int64'))
#
# print("Feature Outliers:", len(all_outliers))

red_table = red_wine_quality_df.describe()
white_table = white_wine_quality_df.describe()




# PCA visulalization

# Get the labels
true_labels = wine_quality_df['type']

# Get all the features for input
tmp_X = wine_quality_df.drop(['type'], axis=1)

# Show the full pca graphs
X_pca = pca(tmp_X, 2)
plot_2d_clusters(X_pca, true_labels, ["white", "red"])
plt.tight_layout()
plt.savefig(figures_dir + "2d_pca.png")

# Show the full pca graph
X_pca = pca(tmp_X, 3)
plot_3d_clusters(X_pca, true_labels, ["white", "red"])
plt.tight_layout()
plt.savefig(figures_dir + "3d_pca.png")





# ### Correlation
print("\n\n-------------------------------------------------------------")
print("Correlation:")

# Generating the correlation matrix between the features
husl_white = rgb_to_husl(250/256, 217/256, 112/256)
husl_red = rgb_to_husl(173/256, 5/256, 64/256)
cmap = sns.diverging_palette(husl_white[0], husl_red[0], s=husl_red[1], l=husl_red[2], as_cmap=True)

tmp_wine_quality_df = wine_quality_df.copy()
tmp_wine_quality_df.loc[:, 'type'].replace({'red':0, 'white': 1}, inplace=True)

print(tmp_wine_quality_df.corr())
#sns.heatmap(tmp_wine_quality_df.corr(), cmap=cmap, square=True, linewidths=.3)
#plt.tight_layout()
#plt.savefig(figures_dir + 'corr.png')

# Generating the AVG correlation value for each feature
avg_corr = pd.DataFrame(tmp_wine_quality_df.corr().abs().mean(axis=0), columns=['avarage absolute correlation']).sort_values(by=['avarage absolute correlation'], ascending=False)
print(avg_corr)
#sns.heatmap(avg_corr, cmap=cmap, linewidths=.3, annot=True)
#plt.plot()
#plt.tight_layout()
#plt.savefig(figures_dir + 'corr_avg.png')

# Exploring the data (by types of wines)
#sns.pairplot(wine_quality_df, hue="type")





# # Arranging Labels and Features

# TYPE CLUSTERING

# Get the labels
print("\n\n-------------------------------------------------------------")
print("Features and Labels:")
true_labels = wine_quality_df['type']
unique_labels = list(set(true_labels))
print("Labels:", unique_labels)
print("Labels Distribution:", dict(zip(*np.unique(true_labels, return_counts=True))))

# Get the features for input
X = wine_quality_df.drop(['type', 'quality', 'alcohol', 'free sulfur dioxide'], axis=1)
print("Features Size:", X.shape)

#X = pca(X, 9)

# Scale the values
X_scaled = StandardScaler().fit_transform(X)
#X_scaled = normalize(X)
#X_scaled = X

# Show the full pca graphs after pre-processing
X_pca = pca(X_scaled, 2)
plot_2d_clusters(X_pca, true_labels, ["white", "red"])
plt.tight_layout()
plt.savefig(figures_dir + "2d_pca_preprocessed.png")

X_pca = pca(X_scaled, 3)
plot_3d_clusters(X_pca, true_labels, ["white", "red"])
plt.tight_layout()
plt.savefig(figures_dir + "3d_pca_preprocessed.png")






# # Clustering Algorithms

# ### GMM

print("\n\n-------------------------------------------------------------")
print("GMM Algorithm:")
gmm_x_scaled = normalize(X_scaled)
original_gmm = GaussianMixture(len(unique_labels), random_state=seed)
gmm = original_gmm.fit(gmm_x_scaled)
preds = gmm.predict(gmm_x_scaled)

print_cv_report(gmm_x_scaled, true_labels, original_gmm, k=10)
noise_indexes = print_score(gmm_x_scaled, true_labels, preds).reshape(-1)
plot_comparison_clusters(gmm_x_scaled, preds, noise_indexes, true_labels, 3, with_noise=False, with_silhouette=False)
plt.savefig(figures_dir + "gmm_results.png")

search_for_best_model_1d(gmm_x_scaled, true_labels,
                         model_fit=lambda n, X: GaussianMixture(n).fit(X),
                         model_predict=lambda model, X: model.predict(X),
                         options=range(2, 10),
                         param_name="n_clusters")
plt.savefig(figures_dir + "gmm_find_best.png")


# ### K-means

# k-means intuition

print("\n\n-------------------------------------------------------------")
print("K-means Algorithm:")
# Comparing the variance for the target clusters using ground truth
X_red = wine_quality_df.loc[wine_quality_df['type'] == "red"].drop(['type', 'quality', 'alcohol', 'free sulfur dioxide'], axis=1)
X_white = wine_quality_df.loc[wine_quality_df['type'] == "white"].drop(['type', 'quality', 'alcohol', 'free sulfur dioxide'], axis=1)
X_red_scaled = StandardScaler().fit_transform(X_red)
X_white_scaled = StandardScaler().fit_transform(X_white)

X_red_variance = np.var(X_red_scaled, axis=0)
X_white_variance = np.var(X_white_scaled, axis=0)
print("Variance Difference:", X_red_variance - X_white_variance)


# k-means
original_kmeans = KMeans(n_clusters=len(unique_labels), random_state=seed)
kmeans = original_kmeans.fit(X_scaled)
preds = kmeans.predict(X_scaled)
centers = kmeans.cluster_centers_

print_cv_report(X_scaled, true_labels, original_kmeans, k=10)
noise_indexes = print_score(X_scaled, true_labels, preds).reshape(-1)
plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3, with_noise=False, with_silhouette=False)
plt.savefig(figures_dir + "kmeans_results.png")

search_for_best_model_1d(X_scaled, true_labels,
                         model_fit=lambda n, X: KMeans(n).fit(X),
                         model_predict=lambda model, X: model.predict(X),
                         options=range(2, 10),
                         param_name="n_clusters")
plt.savefig(figures_dir + "kmeans_find_best.png")


# ### Spectral Clustering

# Spectral Clustering
print("\n\n-------------------------------------------------------------")
print("Spectral Clustering Algorithm:")
original_spectralClustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=8, n_clusters=len(unique_labels), random_state=seed)
spectralClustering = original_spectralClustering.fit(X_scaled)
preds = spectralClustering.labels_

print_cv_report(X_scaled, true_labels, original_spectralClustering, k=10)
noise_indexes = print_score(X_scaled, true_labels, spectralClustering.labels_).reshape(-1)
plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3, with_noise=False, with_silhouette=False)
plt.savefig(figures_dir + "spectral_results.png")

calculate_pvalue(true_labels, preds, 0.9177, normalized_mutual_info_score, "NMI", pvalue_num_of_shuffles, show_graph=True)
plt.savefig(figures_dir + "spectral_pvalue.png")

search_for_best_model_2d(X_scaled, true_labels,
                         model_fit=lambda n_neighbors, n_clusters, X: SpectralClustering(affinity='nearest_neighbors', n_neighbors=8, n_clusters=n_clusters).fit(X),
                         model_predict=lambda model, X: model.labels_,
                         first_options=range(8, 50),
                         second_options=range(2, 10),
                         first_param_name="n_neighbors",
                         second_param_name="n_clusters")
plt.savefig(figures_dir + "spectral_find_best.png")


# Spectral Clustering without pre-processing
print("\nSpectral Clustering Algorithm without Pre-processing:")
X_without_preprocessing = wine_quality_df.drop(['type'], axis=1)
original_spectralClustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=8, n_clusters=len(unique_labels), random_state=seed)
spectralClustering = original_spectralClustering.fit(X_without_preprocessing)
preds = spectralClustering.labels_

print_cv_report(X_without_preprocessing, true_labels, original_spectralClustering, k=10)
print_score(X_without_preprocessing, true_labels, preds).reshape(-1)


# ### Fuzzy C-means

# fuzzy c-means
print("\n\n-------------------------------------------------------------")
print("Fuzzy C-means Algorithm:")
original_fcm = FCM(n_clusters=len(unique_labels), random_state=seed)
fcm = original_fcm.fit(X_scaled)
preds = fcm.u.argmax(axis=1)

#print_cv_report(X_scaled, true_labels, original_fcm, k=10)
noise_indexes = print_score(X_scaled, true_labels, preds)
plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3, with_noise=False, with_silhouette=False)
plt.savefig(figures_dir + "fcmeans_results.png")

search_for_best_model_1d(X_scaled, true_labels,
                         model_fit=lambda n, X: FCM(n_clusters=n).fit(X),
                         model_predict=lambda model, X: model.u.argmax(axis=1),
                         options=range(2, 10),
                         param_name="n_clusters")
plt.savefig(figures_dir + "fcmeans_find_best.png")


# ### DBSCAN

"""
from time import time
t0 = time()
mds = MDS(2, max_iter=10, n_init=1)
trans_data = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))

tempdf =pd.DataFrame(trans_data)
tempdf.head()
mds_df_with_true_label= tempdf.copy()
mds_df_with_true_label["Class"]=true_labels
mds_df_with_true_label = mds_df_with_true_label.rename(columns={0: "Dim_1", 1: "Dim_2"})
mds_df_with_true_label.head()

plt.figure(figsize=(8,4))
sns.scatterplot(x="Dim_1",y="Dim_2",hue="Class",data= mds_df_with_true_label,palette =sns.color_palette("hls",2))
ax = plt.gca()
ax.set_title("Data after MDS dimension reduce")

#from sklearn.neighbors import kneighbors_graph
#A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)
#print(A.toarray()[0])

# dbscan
dbscan = DBSCAN(eps=3.0, min_samples=5)
#dbscan = DBSCAN(ep=10.0, min_samples=10)
#dbscan = DBSCAN(eps=1.99, min_samples=20)

dbscan.fit(X_scaled)
preds = dbscan.labels_
#n_clusters = len(set(preds)) - (1 if -1 in preds else 0)
n_clusters = len(set(preds))
n_noise = list(preds).count(-1)
print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)
print(np.unique(preds, return_counts=True))
#noise_indexes = print_score(X_scaled, true_labels, preds)

# search_for_best_model_2d(X, true_labels,
#                          model_fit=lambda eps, min_samples, X: DBSCAN(eps=eps, min_samples=min_samples).fit(X),
#                          model_predict=lambda model, X: model.labels_,
#                          first_options=np.arange(0.5, 10.0, 0.1),
#                          second_options=range(2, 10),
#                          first_param_name="eps",
#                          second_param_name="min_samples")


#plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3)
#search_for_best_model(GaussianMixture, range(2, 10))

search_for_best_model_2d(X, true_labels,
                         model_fit=lambda eps, min_samples, X: DBSCAN(eps=eps, min_samples=min_samples).fit(X),
                         model_predict=lambda model, X: model.labels_,
                         first_options=np.arange(10.0, 20.0, 0.1),
                         second_options=range(5, 20),
                         first_param_name="eps",
                         second_param_name="min_samples")

search_for_best_model_2d(X, true_labels,
                         model_fit=lambda eps, min_samples, X: DBSCAN(eps=eps, min_samples=min_samples).fit(X),
                         model_predict=lambda model, X: model.labels_,
                         first_options=np.arange(10.0, 12.0, 0.01),
                         second_options=range(5, 30),
                         first_param_name="eps",
                         second_param_name="min_samples")

results = pd.DataFrame(columns=['eps', 'nmi', 'silhouette', 'detailed_groups'])

search_for_best_model_1d(X, true_labels,
                         model_fit=lambda eps, X: DBSCAN(eps=eps, min_samples=5).fit(X),
                         model_predict=lambda model, X: model.labels_,
                         options=np.arange(7.0, 40.0, 0.01))

ns = 20
tmp_X = StandardScaler().fit_transform(X)
nbrs = NearestNeighbors(n_neighbors=ns).fit(tmp_X)
distances, indices = nbrs.kneighbors(tmp_X)
#print(indices[:,0])
#print(distances)
#print(distances[:,ns-1])
#print(list(zip(indices[:,0], distanceDec)))
distanceDec = sorted(distances[:,ns-1], reverse=True)
#print(distanceDec)
#plt.scatter(indices[:,0], distanceDec)

plt.hist(distanceDec, bins=150)
plt.show()

plt.plot(distanceDec, list(range(1,len(X)+1)))
"""






# # Anomaly Detection

# Anomaly Detection on GMM
print("\n\n-------------------------------------------------------------")
print("Anomaly Detection GMM")
gmm_x_scaled = normalize(X_scaled)
original_gmm = GaussianMixture(len(unique_labels), random_state=seed)
gmm = original_gmm.fit(gmm_x_scaled)
preds = gmm.predict(gmm_x_scaled)

noise_indexes = print_score(gmm_x_scaled, true_labels, preds).reshape(-1)
plot_comparison_clusters(gmm_x_scaled, preds, noise_indexes, true_labels, 3, with_noise=True, with_silhouette=True)
plt.savefig(figures_dir + "gmm_anomaly.png")


# Anomaly Detection on k-means
print("\n\n-------------------------------------------------------------")
print("Anomaly Detection k-means")
original_kmeans = KMeans(n_clusters=len(unique_labels), random_state=seed)
kmeans = original_kmeans.fit(X_scaled)
preds = kmeans.predict(X_scaled)

noise_indexes = print_score(X_scaled, true_labels, preds).reshape(-1)
plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3, with_noise=True, with_silhouette=True)
plt.savefig(figures_dir + "kmeans_anomaly.png")


# Anomaly Detection on Spectral Clustering
print("\n\n-------------------------------------------------------------")
print("Anomaly Detection Spectral Clustering")
original_spectralClustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=8, n_clusters=len(unique_labels), random_state=seed)
spectralClustering = original_spectralClustering.fit(X_scaled)
preds = spectralClustering.labels_

noise_indexes = print_score(X_scaled, true_labels, spectralClustering.labels_).reshape(-1)
plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3, with_noise=True, with_silhouette=True)
plt.savefig(figures_dir + "spectral_anomaly.png")