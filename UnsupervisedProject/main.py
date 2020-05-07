#!/usr/bin/env python
# coding: utf-8

# In[416]:


import pickle
import string
import re
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import colorConverter
import seaborn as sns
from fcmeans import FCM
from scipy.cluster.hierarchy import fclusterdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, classification_report, accuracy_score, recall_score, precision_score, f1_score, \
    make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from functools import partial
from itertools import permutations, product
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D

cmap = cm.get_cmap("Spectral")
show_cross_validation_scores = True
# mpl.rcParams ['figure.dpi'] = 200


# # Loading the dataset

# In[2]:


# Loading the wine dataset
red_wine_quality_df = pd.read_csv('../../winequality-red.csv', header=0, encoding='utf-8', delimiter=';')
red_wine_quality_df['type'] = 'red'
white_wine_quality_df = pd.read_csv('../../winequality-white.csv', header=0, encoding='utf-8', delimiter=';')
white_wine_quality_df['type'] = 'white'
wine_quality_df = pd.concat([red_wine_quality_df, white_wine_quality_df]).reset_index(drop=True)
wine_quality_df = pd.DataFrame.sample(wine_quality_df, frac=1).reset_index(drop=True)

wine_quality_df.head()


# # Dimensionality Reduction Functions

# In[3]:


def pca(X, dimension):
    pca = PCA(n_components=dimension)
    principalComponents = pca.fit_transform(X)
    return principalComponents


# In[4]:


def ica(X, dimension):
    ica = FastICA(n_components=dimension)
    principalComponents = ica.fit_transform(X)
    return principalComponents


# In[5]:


def mds(X, dimension):
    mds = MDS(n_components=dimension)
    principalComponents = mds.fit_transform(X)
    return principalComponents


# # Plotting Functions

# In[6]:


def color_by_cluster(cluster_name):
    if cluster_name == "white":
        return np.array([250, 217, 112]) / 255.0
    elif cluster_name == "red":
        return np.array([173, 5, 64]) / 255.0
    elif cluster_name == 0 or cluster_name == "0":
        return np.array([32, 118, 180]) / 255.0
    elif cluster_name == 1 or cluster_name == "1":
        return np.array([254, 127, 15]) / 255.0
    elif cluster_name == 'noise':
        return np.array([48, 131, 44]) / 255.0

    return None


# In[352]:


def plot_2d_clusters(X, labels, clusters, ax=None):
    # PCA to 2 dimensions
    principalComponents = pca(X, 2)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    labelsDf = pd.DataFrame(data=zip(range(len(labels)), labels), columns=['count', 'cluster'])
    finalDf = pd.concat([principalDf, labelsDf], axis=1)

    # plotting the resulted vectors in a figure, clustered according to the predictions
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('2 component PCA', fontsize=20)

    # ax.set_xlabel('Principal Component 1', fontsize = 12)
    # ax.set_ylabel('Principal Component 2', fontsize = 12)

    for cluster in clusters:
        indicesToKeep = finalDf['cluster'] == cluster
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c=color_by_cluster(cluster))

    ax.legend(clusters)
    ax.grid()


# In[381]:


def plot_3d_clusters(X, labels, clusters, ax=None):
    # PCA to 3 dimensions
    principalComponents = pca(X, 3)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2', 'principal component 3'])
    labelsDf = pd.DataFrame(data=zip(range(len(labels)), labels), columns=['count', 'cluster'])
    finalDf = pd.concat([principalDf, labelsDf], axis=1)

    # plotting the resulted vectors in a figure, clustered according to the predictions
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    # ax.set_xlabel('Principal Component 1', fontsize = 12)
    # ax.set_ylabel('Principal Component 2', fontsize = 12)
    # ax.set_zlabel('Principal Component 3', fontsize = 12)
    ax.set_title('3 component PCA', fontsize=20)

    for cluster in clusters:
        indicesToKeep = finalDf['cluster'] == cluster
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   finalDf.loc[indicesToKeep, 'principal component 3'],
                   c=color_by_cluster(cluster))

    ax.legend(clusters)
    ax.grid()


# In[350]:


def silhouette_plot(X, y, clusters, ax=None):
    if ax is None:
        ax = plt.gca()

    # Compute the silhouette scores for each sample
    silhouette_avg = silhouette_score(X, y)
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = padding = 2
    for cluster in clusters:
        # Aggregate the silhouette scores for samples belonging to
        ith_cluster_silhouette_values = sample_silhouette_values[y == cluster]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0,
                         ith_cluster_silhouette_values,
                         facecolor=color_by_cluster(cluster),
                         alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.15, y_lower + 0.5 * size_cluster_i, str(cluster), fontsize=12)

        # Compute the new y_lower for next plot
        y_lower = y_upper + padding

    ax.set_xlabel("The silhouette coefficient values", fontsize=12)
    ax.set_ylabel("Cluster label", fontsize=12)

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, c='r', alpha=0.8, lw=0.8, ls='-')
    ax.annotate('Average',
                xytext=(silhouette_avg, y_lower * 1.025),
                xy=(0, 0),
                ha='center',
                alpha=0.8,
                fontsize=12)

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylim(0, y_upper + 1)
    ax.set_xlim(-0.2, 1.0)
    return ax


# In[390]:


def plot_comparison_clusters(X, preds, noise_indexes, true_labels, plot_dim, with_noise=True, with_silhouette=True):
    if with_noise:
        # Updating the predictions with noise (as an additional cluster)
        preds_with_noise = np.array(['noise' if i in noise_indexes else preds[i] for i in range(len(preds))])
        # preds_with_noise = preds
    else:
        preds_with_noise = preds

    preds_clusters = list(set(preds))
    preds_with_noise_clusters = list(set(preds_with_noise))
    true_clusters = list(set(true_labels))

    if plot_dim == 2:
        projection = None
    elif plot_dim == 3:
        projection = '3d'
    else:
        print(f"Plot dimension value {plot_dim} is illagel, can be either 2 or 3!")
        exit()

    if not with_silhouette:
        fig = plt.figure(figsize=(15, 6.8))
        ax1 = fig.add_subplot(1, 2, 1, projection=projection)
        plot_2d_clusters(X, preds_with_noise, preds_with_noise_clusters, ax1)

        ax2 = fig.add_subplot(1, 2, 2, projection=projection)
        plot_2d_clusters(X, true_labels, true_clusters, ax2)
        fig.tight_layout(pad=3.0)

        # ax2 = fig.add_subplot(1,2,2)
        # silhouette_plot(X_scaled, preds, preds_clusters, ax=ax2)
    else:
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(2, 2, 1, projection=projection)
        ax2 = fig.add_subplot(2, 2, 2, projection=projection)
        plot_2d_clusters(X, preds_with_noise, preds_with_noise_clusters, ax1)
        plot_2d_clusters(X, true_labels, true_clusters, ax2)

        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        silhouette_plot(X_scaled, preds, preds_clusters, ax=ax3)
        silhouette_plot(X_scaled, true_labels, unique_labels, ax=ax4)

        ax1.set_title('Predictions', fontsize=20)
        ax2.set_title('Ground Truth', fontsize=20)


# # Exploring The Dataset

# In[383]:


# Get the labels
true_labels = wine_quality_df['type']
unique_labels = list(set(true_labels))
print("Labels:", unique_labels)
print("Labels Distribution:", dict(zip(*np.unique(true_labels, return_counts=True))))

# Get all the features for input
X = wine_quality_df.drop(['type'], axis=1)

# %matplotlib inline
# X_pca = pca(X, 2)
# plot_2d_clusters(X_pca, true_labels, ["white", "red"])
# plt.savefig("images/2d_pca.png")

# Show the full pca graph
# %matplotlib notebook

# X_pca = pca(X, 3)
# plot_3d_clusters(X_pca, true_labels, ["white", "red"])
# plt.savefig("images/3d_pca.png")

# Show the full pca graph

"""
X_pca = pca(X_scaled, 2)
plot_2d_clusters(X_pca, true_labels, ["white", "red"])
plt.savefig("images/updated_2d_pca.png")

X_pca = pca(X_scaled, 3)
plot_3d_clusters(X_pca, true_labels, ["white", "red"])
plt.savefig("images/updated_3d_pca.png")
"""

# In[107]:


from husl import rgb_to_husl

husl_white = rgb_to_husl(250 / 256, 217 / 256, 112 / 256)
husl_red = rgb_to_husl(173 / 256, 5 / 256, 64 / 256)
cmap = sns.diverging_palette(husl_white[0], husl_red[0], s=husl_red[1], l=husl_red[2], as_cmap=True)

tmp_df = wine_quality_df.copy()
sns.heatmap(tmp_df.corr(), cmap=cmap, square=True, linewidths=.3)
plt.tight_layout()
# plt.savefig('corr.png')


# In[12]:


# Exploring the data (by quality values)
f, ax = plt.subplots(12, 3, figsize=(15, 45))
sns.boxplot('quality', 'fixed acidity', data=white_wine_quality_df, ax=ax[0, 0])
sns.boxplot('quality', 'fixed acidity', data=red_wine_quality_df, ax=ax[0, 1])
sns.boxplot('quality', 'fixed acidity', data=wine_quality_df, ax=ax[0, 2])

sns.boxplot('quality', 'sulphates', data=red_wine_quality_df, ax=ax[1, 0])
sns.boxplot('quality', 'sulphates', data=white_wine_quality_df, ax=ax[1, 1])
sns.boxplot('quality', 'sulphates', data=wine_quality_df, ax=ax[1, 2])

sns.boxplot('quality', 'volatile acidity', data=red_wine_quality_df, ax=ax[2, 0])
sns.boxplot('quality', 'volatile acidity', data=white_wine_quality_df, ax=ax[2, 1])
sns.boxplot('quality', 'volatile acidity', data=wine_quality_df, ax=ax[2, 2])

sns.boxplot('quality', 'residual sugar', data=red_wine_quality_df, ax=ax[3, 0])
sns.boxplot('quality', 'residual sugar', data=white_wine_quality_df, ax=ax[3, 1])
sns.boxplot('quality', 'residual sugar', data=wine_quality_df, ax=ax[3, 2])

sns.boxplot('quality', 'chlorides', data=red_wine_quality_df, ax=ax[4, 0])
sns.boxplot('quality', 'chlorides', data=white_wine_quality_df, ax=ax[4, 1])
sns.boxplot('quality', 'chlorides', data=wine_quality_df, ax=ax[4, 2])

sns.boxplot('quality', 'free sulfur dioxide', data=red_wine_quality_df, ax=ax[5, 0])
sns.boxplot('quality', 'free sulfur dioxide', data=white_wine_quality_df, ax=ax[5, 1])
sns.boxplot('quality', 'free sulfur dioxide', data=wine_quality_df, ax=ax[5, 2])

sns.boxplot('quality', 'total sulfur dioxide', data=red_wine_quality_df, ax=ax[6, 0])
sns.boxplot('quality', 'total sulfur dioxide', data=white_wine_quality_df, ax=ax[6, 1])
sns.boxplot('quality', 'total sulfur dioxide', data=wine_quality_df, ax=ax[6, 2])

sns.boxplot('quality', 'density', data=red_wine_quality_df, ax=ax[7, 0])
sns.boxplot('quality', 'density', data=white_wine_quality_df, ax=ax[7, 1])
sns.boxplot('quality', 'density', data=wine_quality_df, ax=ax[7, 2])

sns.boxplot('quality', 'pH', data=red_wine_quality_df, ax=ax[8, 0])
sns.boxplot('quality', 'pH', data=white_wine_quality_df, ax=ax[8, 1])
sns.boxplot('quality', 'pH', data=wine_quality_df, ax=ax[8, 2])

sns.boxplot('quality', 'sulphates', data=red_wine_quality_df, ax=ax[9, 0])
sns.boxplot('quality', 'sulphates', data=white_wine_quality_df, ax=ax[9, 1])
sns.boxplot('quality', 'sulphates', data=wine_quality_df, ax=ax[9, 2])

sns.boxplot('quality', 'alcohol', data=red_wine_quality_df, ax=ax[10, 0])
sns.boxplot('quality', 'alcohol', data=white_wine_quality_df, ax=ax[10, 1])
sns.boxplot('quality', 'alcohol', data=wine_quality_df, ax=ax[10, 2])

sns.boxplot('quality', 'quality', data=red_wine_quality_df, ax=ax[11, 0])
sns.boxplot('quality', 'quality', data=white_wine_quality_df, ax=ax[11, 1])
sns.boxplot('quality', 'quality', data=wine_quality_df, ax=ax[11, 2])

print("\t\t    Red\t\t\t\t     White\t\t\t\t Total")

# In[13]:


# Calculating the correlation of one feature to the others
target_feature = 'quality'
print("RED:\n")
print(red_wine_quality_df.corr(method='pearson')[target_feature].sort_values(ascending=False))
print("\nWHITE:\n")
print(white_wine_quality_df.corr(method='pearson')[target_feature].sort_values(ascending=False))
print("\nTOTAL:\n")
print(wine_quality_df.corr(method='pearson')[target_feature].sort_values(ascending=False))

# In[56]:


# Exploring the data (by types of wines)
f, ax = plt.subplots(4, 3, figsize=(15, 20))
sns.boxplot('type', 'fixed acidity', data=wine_quality_df, ax=ax[0, 0])
sns.boxplot('type', 'sulphates', data=wine_quality_df, ax=ax[0, 1])
sns.boxplot('type', 'volatile acidity', data=wine_quality_df, ax=ax[0, 2])
sns.boxplot('type', 'residual sugar', data=wine_quality_df, ax=ax[1, 0])
sns.boxplot('type', 'chlorides', data=wine_quality_df, ax=ax[1, 1])
sns.boxplot('type', 'free sulfur dioxide', data=wine_quality_df, ax=ax[1, 2])
sns.boxplot('type', 'total sulfur dioxide', data=wine_quality_df, ax=ax[2, 0])
sns.boxplot('type', 'density', data=wine_quality_df, ax=ax[2, 1])
sns.boxplot('type', 'pH', data=wine_quality_df, ax=ax[2, 2])
sns.boxplot('type', 'sulphates', data=wine_quality_df, ax=ax[3, 0])
sns.boxplot('type', 'alcohol', data=wine_quality_df, ax=ax[3, 1])
sns.boxplot('type', 'quality', data=wine_quality_df, ax=ax[3, 2])

tmp_df = wine_quality_df.drop(["type"], axis=1)
all_outliers = np.array([], dtype='int64')
for feature in tmp_df.keys():
    Q1 = np.percentile(tmp_df[feature], 25)
    Q3 = np.percentile(tmp_df[feature], 75)
    step = 1.5 * (Q3 - Q1)
    outlier_pts = tmp_df[~((tmp_df[feature] >= Q1 - step) & (tmp_df[feature] <= Q3 + step))]
    all_outliers = np.append(all_outliers, outlier_pts.index.values.astype('int64'))

print("Feature Outliers:", len(all_outliers))

# In[15]:


red_table = red_wine_quality_df.describe()
red_table

# In[16]:


white_table = white_wine_quality_df.describe()
white_table

# In[17]:


red_table - white_table

# In[18]:


sns.pairplot(wine_quality_df, hue="type")
plt.show()


# # Finding The Best Number of clusters

# In[418]:


def scores_for_all_options(X, true_labels, model_fit, model_predict, num_of_options, options):
    wcss_scores = np.zeros(num_of_options)  # Within-Cluster-Sum-of-Squares
    aic_scores = np.zeros(num_of_options)
    bic_scores = np.zeros(num_of_options)
    silhouette_scores = np.zeros(num_of_options)
    nmi_scores = np.zeros(num_of_options)

    if type(options) == tuple:
        options_indexes = list(product(np.arange(0, num_of_options[0]), np.arange(0, num_of_options[1])))
    else:
        options_indexes = np.arange(0, num_of_options)

    pbar = tqdm(options_indexes)
    for option_index in pbar:
        if type(options) == tuple:
            option = (options[0][option_index[0]], options[1][option_index[1]])
            model = model_fit(*option, X)
        else:
            option = options[option_index]
            model = model_fit(option, X)

        preds = model_predict(model, X)

        # Avoiding large number of clusters (for DBSCAN)
        if len(np.unique(preds)) < 3 or len(np.unique(preds)) > 8:
            continue

        if isinstance(model, KMeans):
            wcss_scores[option_index] = model.inertia_
        elif isinstance(model, GaussianMixture):
            aic_scores[option_index] = model.aic(X)
            bic_scores[option_index] = model.bic(X)

        silhouette_scores[option_index] = silhouette_score(X, preds)
        nmi_scores[option_index] = normalized_mutual_info_score(preds, true_labels)
        pbar.set_description(f"NMI={nmi_scores[option_index]:.4}, silhouette={silhouette_scores[option_index]:.4}")

    return wcss_scores, aic_scores, bic_scores, silhouette_scores, nmi_scores


# Searching for the best model based on elbow (for kmeans), silhoette and MI scores
def search_for_best_model_1d(X, true_labels, model_fit, model_predict, options, param_name=""):
    # Defining the scores
    scores = scores_for_all_options(X, true_labels, model_fit, model_predict, len(options), options)
    wcss_scores, aic_scores, bic_scores, silhouette_scores, nmi_scores = scores

    if np.any(wcss_scores != 0):
        fig = plt.figure(figsize=(20, 5))
        n_rows, n_cols = 1, 3
        ax = fig.add_subplot(n_rows, n_cols, 3)

        wcss_plot = sns.lineplot(options, wcss_scores, ax=ax)
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel("WCSS", fontsize=12)

    elif np.any(aic_scores != 0):
        fig = plt.figure(figsize=(20, 5))
        n_rows, n_cols = 1, 3
        ax = fig.add_subplot(n_rows, n_cols, 3)

        ic_plot = sns.lineplot(options, aic_scores, ax=ax, label="aic")
        sns.lineplot(options, bic_scores, ax=ax, label="bic")
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel("IC", fontsize=12)

        ax.legend(fontsize=12)
    else:
        fig = plt.figure(figsize=(15, 5))
        n_rows, n_cols = 1, 2

    ax = fig.add_subplot(n_rows, n_cols, 1)
    silhouette_plot = sns.lineplot(options, silhouette_scores, ax=ax)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Silhouette", fontsize=12)

    ax = fig.add_subplot(n_rows, n_cols, 2)
    mi_plot = sns.lineplot(options, nmi_scores, ax=ax)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("NMI [bits]", fontsize=12)

    # mi_plot.legend(fontsize = 12)
    f.tight_layout(pad=3.0)


# Searching for the best model based on elbow (for kmeans), silhoette and MI scores
def search_for_best_model_2d(X, true_labels, model_fit, model_predict, first_options, second_options,
                             first_param_name="", second_param_name=""):
    num_of_options = (len(first_options), len(second_options))
    X_options, Y_options = np.meshgrid(first_options, second_options)
    X_options = np.transpose(X_options)
    Y_options = np.transpose(Y_options)

    # Defining the scores
    scores = scores_for_all_options(X, true_labels, model_fit, model_predict, num_of_options,
                                    (first_options, second_options))
    wcss_scores, aic_scores, bic_scores, silhouette_scores, nmi_scores = scores

    if np.any(wcss_scores != 0):
        fig = plt.figure(figsize=(15, 5))
        n_rows, n_cols = 1, 3
        ax = fig.add_subplot(n_rows, n_cols, 3, projection='3d')

        wcss_plot = ax.plot_surface(X_options, Y_options, wcss_scores)
        ax.set_xlabel(first_param_name, fontsize=12)
        ax.set_ylabel(second_param_name, fontsize=12)
        ax.set_zlabel("WCSS", fontsize=12)

    elif np.any(aic_scores != 0):
        fig = plt.figure(figsize=(15, 5))
        n_rows, n_cols = 1, 3
        ax = fig.add_subplot(n_rows, n_cols, 3, projection='3d')

        ic_plot = ax.plot_surface(X_options, Y_options, aic_scores, label="aic")
        ic_plot = ax.plot_surface(X_options, Y_options, bic_scores, label="bic")
        ax.set_xlabel(first_param_name, fontsize=12)
        ax.set_ylabel(second_param_name, fontsize=12)
        ax.set_zlabel("IC", fontsize=12)
        ax.legend(fontsize=12)
    else:
        fig = plt.figure(figsize=(10, 5))
        n_rows, n_cols = 1, 2

    ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    silhouette_plot = ax.plot_surface(X_options, Y_options, silhouette_scores)
    ax.set_xlabel(first_param_name, fontsize=12)
    ax.set_ylabel(second_param_name, fontsize=12)
    ax.set_zlabel("Silhouette", fontsize=12)

    ax = fig.add_subplot(n_rows, n_cols, 2, projection='3d')
    mi_plot = ax.plot_surface(X_options, Y_options, nmi_scores)
    ax.set_xlabel(first_param_name, fontsize=12)
    ax.set_ylabel(second_param_name, fontsize=12)
    ax.set_zlabel("NMI [bits]", fontsize=12)

    f.tight_layout(pad=3.0)


# # Score Methods

# In[15]:


def calculate_pvalue(true_labels, preds, resulted_score):
    nmi_of_shuffles = np.array(
        [normalized_mutual_info_score(pd.DataFrame.sample(true_labels, frac=1).reset_index(drop=True), preds) for i in
         range(10000)])
    return len(np.where(nmi_of_shuffles > resulted_score)[0]) / len(nmi_of_shuffles)


# ### Clustering Scores

# In[16]:


silhouette_noise_limit = 0


def match_labels(true_labels, preds):
    # Returns new predictions that have match labels to the true labels
    # Each predicted label gets the most suitable label from the true labels

    true_labels = np.array(true_labels)
    new_preds = np.array(true_labels.copy())
    all_pairs = np.dstack((preds, true_labels))[0]
    unique_true_labels = np.unique(true_labels)
    unique_preds = np.unique(preds)
    best_matching = list(permutations(unique_preds, len(unique_preds)))[0]

    if len(unique_true_labels) == len(unique_preds):
        best_preds = new_preds
        best_score = 0
        for permutation in permutations(unique_preds, len(unique_preds)):
            # print(permutation, unique_true_labels)
            new_preds = np.array(true_labels.copy())
            for label_index in range(len(permutation)):
                np.put(new_preds, np.where(preds == permutation[label_index])[0], unique_true_labels[label_index])

            curr_score = f1_score(true_labels, new_preds, average='macro')
            if curr_score > best_score:
                best_score = curr_score
                best_preds = new_preds
                best_matching = permutation
            # print(curr_score)

        # print(best_score)
        new_preds = best_preds
    else:
        for label in np.unique(preds):
            specific_pairs, specific_counts = np.unique(all_pairs[np.where(all_pairs[:, 0] == label)].astype("<U22"),
                                                        axis=0, return_counts=True)
            best_match_index = np.argmax(specific_counts)
            np.put(new_preds, np.where(preds == label)[0], specific_pairs[best_match_index][1])

    return true_labels, new_preds, best_matching


def calculate_scores(X, true_labels, preds):
    silhouette_scores = silhouette_samples(X, preds)
    silhouette_total_score = silhouette_score(X, preds)

    mi_score = mutual_info_score(true_labels, preds)
    ami_score = adjusted_mutual_info_score(true_labels, preds)
    nmi_score = normalized_mutual_info_score(true_labels, preds)

    print(
        f"Silhouette score: Best={max(silhouette_scores):.4}, Wrost={min(silhouette_scores):.4}, Total={silhouette_total_score:.4}")
    print(f"MI scores: Standard={mi_score:.4}, Adjusted={ami_score:.4}, Normalized={nmi_score:.4}")

    true_labels, preds, matching = match_labels(true_labels, preds)

    acc = np.sum([1 for i in range(len(preds)) if true_labels[i] == preds[i]]) / len(preds)
    recall = recall_score(true_labels, preds, average='macro')
    precision = precision_score(true_labels, preds, average='macro')
    f1 = f1_score(true_labels, preds, average='macro')

    confusion_mat = pd.DataFrame(confusion_matrix(true_labels, preds, labels=np.unique(true_labels)))
    confusion_mat.columns = np.unique(true_labels)

    print(f"Precision: {precision:.4}, Recall: {recall:.4}, f1-score: {f1:.4}, Accuracy: {acc:.4}")
    print(confusion_mat)
    # print(classification_report(true_labels, preds))


def get_without_noise(data, noise_indexes):
    if type(data) == np.ndarray:
        return data[np.setdiff1d(np.arange(data.shape[0]), noise_indexes)]
    else:
        return data.iloc[np.setdiff1d(np.arange(data.shape[0]), noise_indexes)].reset_index(drop=True)


def print_score(X, true_labels, preds):
    silhouette_scores = silhouette_samples(X, preds)

    # Calculate the scores with the noise
    print("Full Clustering Scores:")
    calculate_scores(X, true_labels, preds)

    """
    print("\nScores after shuffle:")
    true_labels_shuffled = pd.DataFrame.sample(true_labels, frac=1).reset_index(drop=True)
    calculate_scores(X, true_labels_shuffled, preds)
    """

    # Get the index of noise points (currenty according to silhouette)
    noise_indexes = np.argwhere(silhouette_scores < silhouette_noise_limit).reshape(-1)
    print("\nNumber of NOISE points:", len(noise_indexes))

    # Remove the noise
    X_no_noise = get_without_noise(X, noise_indexes)
    preds_no_noise = get_without_noise(preds, noise_indexes)
    true_labels_no_noise = get_without_noise(true_labels, noise_indexes)

    # Calculate the scores without the noise
    print("\nClustering Scores Without NOISE:")
    calculate_scores(X_no_noise, true_labels_no_noise, preds_no_noise)

    return noise_indexes


# ### Cross Validation Scores

# In[17]:


def cross_val_scorer(model, X, true_labels, score_func):
    if isinstance(model, FCM):
        preds = model.u.argmax(axis=1)
    elif isinstance(model, SpectralClustering):
        preds = model.fit_predict(X)
    else:
        preds = model.predict(X)

    true_labes, preds, _ = match_labels(true_labels, preds)

    if len(set(preds)) != len(set(true_labels)):
        print("The number of target labels isn't equal to the number of true labels!")
        return

    return score_func(true_labels, preds)


def support_score(true_labels, preds, pos_label=None, average=None):
    if pos_label == None:
        return len(true_labels)
    else:
        return len(np.where(true_labels == pos_label)[0])


def print_cv_report(X, true_labels, model, k=10):
    if not show_cross_validation_scores:
        return

    print("Cross Validation Report:")

    unique_labels = list(set(true_labels))

    # (score_name, score_func, another_score)
    scores = [("accuracy", accuracy_score, False),
              ("precision", precision_score, True),
              ("recall", recall_score, True),
              ("f1-score", f1_score, True),
              ("support", support_score, True)]

    average_options = ["macro", "micro", "weighted"]
    average_names = ["macro avg", "micro avg", "weighted avg"]

    scorers = {}
    for score_name, score_func, another_label in scores:
        if another_label == False:
            scorers[score_name] = partial(cross_val_scorer, score_func=partial(score_func))
            scorers["support_" + score_name] = partial(cross_val_scorer, score_func=partial(support_score))
        else:
            for label in unique_labels:
                scorers[score_name + "_" + str(label)] = partial(cross_val_scorer,
                                                                 score_func=partial(score_func, pos_label=label))

            for average in average_options:
                scorers[score_name + "_" + average] = partial(cross_val_scorer,
                                                              score_func=partial(score_func, average=average))

    cross_val_results = cross_validate(model, X, true_labels, scoring=scorers, cv=k)
    cross_val_dict = {}

    for row_type in unique_labels + average_options:
        if row_type in average_options:
            row_name = row_type + " avg"
        else:
            row_name = row_type

        cross_val_dict[row_name] = {}
        for score_name, _, _ in scores:
            if score_name != "accuracy":
                cross_val_dict[row_name][score_name] = np.mean(cross_val_results["test_" + score_name + "_" + row_type])
            else:
                cross_val_dict[score_name] = {"f1-score": np.mean(cross_val_results["test_" + score_name])}
                cross_val_dict[score_name]["support"] = np.mean(cross_val_results["test_support_accuracy"])

    cross_validation_report = pd.DataFrame(cross_val_dict).transpose()
    cross_validation_report = cross_validation_report.reindex(unique_labels + ["accuracy"] + average_names).round(4)

    print(cross_validation_report)
    print("")


# # Arranging Labels and Features

# In[18]:


# QUALITY CLUSTERING

tmp_wine_quality = red_wine_quality_df.copy()
tmp_wine_quality['is average?'] = tmp_wine_quality.apply(lambda row: row['quality'] >= 4 and row['quality'] <= 6,
                                                         axis=1)
tmp_wine_quality = tmp_wine_quality.loc[
    (tmp_wine_quality['quality'] <= 5) | (tmp_wine_quality['quality'] >= 6)].reset_index(drop=True)
tmp_wine_quality['is great?'] = tmp_wine_quality.apply(lambda row: row['quality'] > 5, axis=1)
# tmp_wine_quality['is great?'] = tmp_wine_quality.apply(lambda row: row['quality'] > 5, axis=1)

tmp_wine_quality = tmp_wine_quality.reset_index(drop=True)

# Get the labels
# true_labels = tmp_wine_quality['is average?']
true_labels = tmp_wine_quality['is great?']
# true_labels = red_wine_quality_df['quality']
# true_labels = wine_quality_df['quality']
unique_labels = list(set(true_labels))
print("Labels: ", unique_labels)
print("Labels Distribution:", dict(zip(*np.unique(true_labels, return_counts=True))))

# Get the features for input
# X = tmp_wine_quality.drop(['type', 'is average?', 'is great?', 'quality'], axis=1).loc[:, ['alcohol', 'sulphates', 'citric acid', 'volatile acidity', 'density']]
X = tmp_wine_quality.copy()
X['0'] = X['alcohol']
X['1'] = 2 * X['alcohol']
X = X.loc[:, ['0', '1']]
# X['alcohol'] *= 1000000
# X = tmp_wine_quality.drop(['type', 'is average?', 'is great?', 'quality'], axis=1).loc[:, ['alcohol', 'total sulfur dioxide','citric acid']]
# X = red_wine_quality_df.drop(['quality', 'type'], axis=1)
print("Features Size:", X.shape)

# X = pca(X, 4)

# Scale the values
# X_scaled = StandardScaler().fit_transform(X)
# X_scaled = normalize(X)
X_scaled = X

# X_lower = mds(X, 3)
# print(X_lower)
# print(X_scaled)
# %matplotlib notebook
# plot_2d_clusters(X, true_labels, unique_labels)


# In[19]:


# %matplotlib inline
# sns.pairplot(tmp_wine_quality, hue="is great?")
# sns.pairplot(tmp_wine_quality, hue="density")
# plt.show()


# In[20]:


# ALCOHOL CLUSTERING

tmp_wine_quality = red_wine_quality_df.copy()
tmp_wine_quality['is a lot of alcohol?'] = tmp_wine_quality.apply(lambda row: row['alcohol'] >= 11.0, axis=1)
tmp_wine_quality = tmp_wine_quality.reset_index(drop=True)

# Get the labels
# true_labels = tmp_wine_quality['is average?']
true_labels = tmp_wine_quality['is a lot of alcohol?']
# true_labels = red_wine_quality_df['quality']
# true_labels = wine_quality_df['quality']
unique_labels = list(set(true_labels))
print("Labels: ", unique_labels)
print("Labels Distribution:", dict(zip(*np.unique(true_labels, return_counts=True))))

# Get the features for input
X = tmp_wine_quality.drop(['type', 'is a lot of alcohol?', 'quality', 'alcohol'], axis=1)
# X = tmp_wine_quality.drop(['type', 'is average?', 'is great?', 'quality'], axis=1).loc[:, ['alcohol', 'sulphates', 'citric acid']]
# X = wine_quality_df.drop(['quality', 'type'], axis=1)
print("Features Size:", X.shape)

# X = pca(X, 4)

# Scale the values
X_scaled = StandardScaler().fit_transform(X)
# X_scaled = normalize(X)
# X_scaled = X

# X_lower = mds(X, 3)
# print(X_lower)
# %matplotlib notebook
# plot_3d_clusters(X, true_labels, unique_labels)


# In[361]:


# TYPE CLUSTERING

# Get the labels
true_labels = wine_quality_df['type']
unique_labels = list(set(true_labels))
print("Labels:", unique_labels)
print("Labels Distribution:", dict(zip(*np.unique(true_labels, return_counts=True))))

# Get the features for input
X = wine_quality_df.drop(['type', 'quality', 'alcohol', 'free sulfur dioxide'], axis=1)
print("Features Size:", X.shape)

# X = pca(X, 9)

# Scale the values
X_scaled = StandardScaler().fit_transform(X)
# X_scaled = normalize(X)
# X_scaled = X


# # Clustering Algorithms

# ### K-means

# In[391]:


# k-means
original_kmeans = KMeans(n_clusters=len(unique_labels))
kmeans = original_kmeans.fit(X_scaled)
preds = kmeans.predict(X_scaled)
centers = kmeans.cluster_centers_
n_clusters = len(centers)
# plot_confusion_matrix(true_labels, preds)
# print(calculate_pvalue(true_labels, preds, 0.1))
print_cv_report(X_scaled, true_labels, original_kmeans, k=10)
noise_indexes = print_score(X_scaled, true_labels, preds).reshape(-1)
plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3, with_noise=False, with_silhouette=False)
plt.savefig("images/kmeans_results.png")

search_for_best_model_1d(X_scaled, true_labels,
                         model_fit=lambda n, X: KMeans(n).fit(X),
                         model_predict=lambda model, X: model.predict(X),
                         options=range(2, 10),
                         param_name="n_clusters")
plt.savefig("images/kmeans_find_best.png")

# ### GMM

# In[392]:


# GMM
gmm_x_scaled = normalize(X_scaled)
original_gmm = GaussianMixture(len(unique_labels))
gmm = original_gmm.fit(gmm_x_scaled)
preds = gmm.predict(gmm_x_scaled)
n_clusters = gmm.n_components

print_cv_report(gmm_x_scaled, true_labels, original_gmm, k=10)
noise_indexes = print_score(gmm_x_scaled, true_labels, preds).reshape(-1)
plot_comparison_clusters(gmm_x_scaled, preds, noise_indexes, true_labels, 3, with_noise=False, with_silhouette=False)
plt.savefig("images/gmm_results.png")

search_for_best_model_1d(gmm_x_scaled, true_labels,
                         model_fit=lambda n, X: GaussianMixture(n).fit(X),
                         model_predict=lambda model, X: model.predict(X),
                         options=range(2, 10),
                         param_name="n_clusters")
plt.savefig("images/gmm_find_best.png")

# ### Fuzzy C-means

# In[393]:


# fuzzy c-means
original_fcm = FCM(n_clusters=len(unique_labels))
fcm = original_fcm.fit(X_scaled)
centers = fcm.centers
n_clusters = len(centers)
preds = fcm.u.argmax(axis=1)

# print_cv_report(X_scaled, true_labels, original_fcm, k=10)
noise_indexes = print_score(X_scaled, true_labels, preds)
plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3, with_noise=False, with_silhouette=False)
plt.savefig("images/fcmeans_results_figure.png")

search_for_best_model_1d(X_scaled, true_labels,
                         model_fit=lambda n, X: FCM(n_clusters=n).fit(X),
                         model_predict=lambda model, X: model.u.argmax(axis=1),
                         options=range(2, 10),
                         param_name="n_clusters")
plt.savefig("images/fcmeans_find_best_figure.png")

# ### DBSCAN

# In[128]:


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
"""

# In[419]:


# from sklearn.neighbors import kneighbors_graph
# A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)
# print(A.toarray()[0])

# dbscan
# dbscan = DBSCAN(eps=3.0, min_samples=5)
# dbscan = DBSCAN(eps=20, min_samples=10)
# dbscan.fit(X)
# preds = dbscan.labels_
# n_clusters = len(set(preds)) - (1 if -1 in preds else 0)
# n_clusters = len(set(preds))
# n_noise = list(preds).count(-1)
# print("Number of clusters:", n_clusters)
# print("Number of noise points:", n_noise)
# noise_indexes = print_score(X_scaled, true_labels, preds)

search_for_best_model_2d(X, true_labels,
                         model_fit=lambda eps, min_samples, X: DBSCAN(eps=eps, min_samples=min_samples).fit(X),
                         model_predict=lambda model, X: model.labels_,
                         first_options=np.arange(0.5, 10.0, 0.1),
                         second_options=range(2, 10),
                         first_param_name="eps",
                         second_param_name="min_samples")

# plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3)
# search_for_best_model(GaussianMixture, range(2, 10))


# In[420]:


search_for_best_model_2d(X, true_labels,
                         model_fit=lambda eps, min_samples, X: DBSCAN(eps=eps, min_samples=min_samples).fit(X),
                         model_predict=lambda model, X: model.labels_,
                         first_options=np.arange(10.0, 20.0, 0.1),
                         second_options=range(5, 20),
                         first_param_name="eps",
                         second_param_name="min_samples")

# In[ ]:


search_for_best_model_2d(X, true_labels,
                         model_fit=lambda eps, min_samples, X: DBSCAN(eps=eps, min_samples=min_samples).fit(X),
                         model_predict=lambda model, X: model.labels_,
                         first_options=np.arange(10.0, 12.0, 0.01),
                         second_options=range(5, 30),
                         first_param_name="eps",
                         second_param_name="min_samples")

# ### Spectral Clustering

# In[388]:


# Spectral Clustering
original_spectralClustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=8,
                                                 n_clusters=len(unique_labels))
spectralClustering = original_spectralClustering.fit(X_scaled)
preds = spectralClustering.labels_
n_clusters = len(unique_labels)

print_cv_report(X_scaled, true_labels, original_spectralClustering, k=10)
noise_indexes = print_score(X_scaled, true_labels, spectralClustering.labels_).reshape(-1)
plot_comparison_clusters(X_scaled, preds, noise_indexes, true_labels, 3, with_noise=False, only_preds=True)
plt.savefig("images/spectral_results_figure.png")

search_for_best_model_2d(X_scaled, true_labels,
                         model_fit=lambda n_neighbors, n_clusters, X: SpectralClustering(affinity='nearest_neighbors',
                                                                                         n_neighbors=8,
                                                                                         n_clusters=n_clusters).fit(X),
                         model_predict=lambda model, X: model.labels_,
                         first_options=range(8, 50),
                         second_options=range(2, 10),
                         first_param_name="n_neighbors",
                         second_param_name="n_clusters")
plt.savefig("images/spectral_find_best_figure.png")

