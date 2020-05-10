import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
from fcmeans import FCM
from scipy.cluster.hierarchy import fclusterdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples, mutual_info_score, normalized_mutual_info_score, classification_report, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from functools import partial
from itertools import permutations, product
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D
from husl import rgb_to_husl
from matplotlib.colors import colorConverter

import warnings
warnings.filterwarnings("ignore")
mpl.use("Agg")  # Prevent showing stuff
#mpl.rcParams ['figure.dpi'] = 200
cmap = cm.get_cmap("Spectral")

# Constants and Flags
show_cross_validation_scores = True
figures_dir = "figures/"
pvalue_num_of_shuffles = 10000
silhouette_noise_limit = 0
pvalue_flag = False
seed = 42




# # Dimensionality Reduction Functions

def pca(X, dimension):
	pca = PCA(n_components=dimension)
	principalComponents = pca.fit_transform(X)
	return principalComponents

def ica(X, dimension):
	ica = FastICA(n_components=dimension)
	principalComponents = ica.fit_transform(X)
	return principalComponents


def mds(X, dimension):
	mds = MDS(n_components=dimension)
	principalComponents = mds.fit_transform(X)
	return principalComponents





# # Plotting Functions

def color_by_cluster(cluster_name):
	if cluster_name == "white":
		return np.array([250, 217, 112]) / 255.0
	elif cluster_name == "red":
		return np.array([173, 5, 64]) / 255.0
	elif cluster_name == 0 or cluster_name == "0":
		return np.array([174, 125, 204]) / 255.0
	elif cluster_name == 1 or cluster_name == "1":
		return np.array([148, 189, 67]) / 255.0
	elif cluster_name == 'noise':
		return np.array([0, 0, 0]) / 255.0

	#     if cluster_name == "white":
	#         return np.array([250, 217, 112]) / 255.0
	#     elif cluster_name == "red":
	#         return np.array([173, 5, 64]) / 255.0
	#     elif cluster_name == 0 or cluster_name == "0":
	#         return np.array([32, 118, 180]) / 255.0
	#     elif cluster_name == 1 or cluster_name == "1":
	#         return np.array([254, 127, 15]) / 255.0
	#     elif cluster_name == 'noise':
	#         return np.array([48, 131, 44]) / 255.0

	return None

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

def plot_comparison_clusters(X, preds, noise_indexes, true_labels, plot_dim, with_noise=True, with_silhouette=True,
							 full=False):
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

	if full:
		# Show both silhouette and 3d PCA graphs for preds and ground truth
		fig = plt.figure(figsize=(15, 15))
		ax1 = fig.add_subplot(2, 2, 1, projection=projection)
		ax2 = fig.add_subplot(2, 2, 2, projection=projection)
		plot_2d_clusters(X, preds_with_noise, preds_with_noise_clusters, ax1)
		plot_2d_clusters(X, true_labels, true_clusters, ax2)
		ax1.set_title('Predictions', fontsize=20)
		ax2.set_title('Ground Truth', fontsize=20)

		ax3 = fig.add_subplot(2, 2, 3)
		ax4 = fig.add_subplot(2, 2, 4)
		silhouette_plot(X, preds, preds_clusters, ax=ax3)
		silhouette_plot(X, true_labels, np.unique(true_labels), ax=ax4)

	elif not with_silhouette:
		# Show only 3d PCA graphs for preds and ground truth
		fig = plt.figure(figsize=(15, 6.8))
		ax1 = fig.add_subplot(1, 2, 1, projection=projection)
		plot_2d_clusters(X, preds_with_noise, preds_with_noise_clusters, ax1)
		ax1.set_title('Predictions', fontsize=20)

		ax2 = fig.add_subplot(1, 2, 2, projection=projection)
		plot_2d_clusters(X, true_labels, true_clusters, ax2)
		fig.tight_layout(pad=3.0)
		ax2.set_title('Ground Truth', fontsize=20)

	else:  # with_silhouette
		# Show silhouette and 3d PCA graphs for the preds only
		fig = plt.figure(figsize=(15, 6.8))
		ax1 = fig.add_subplot(1, 2, 1, projection=projection)
		plot_2d_clusters(X, preds_with_noise, preds_with_noise_clusters, ax1)
		ax1.set_title('Predictions', fontsize=20)

		ax2 = fig.add_subplot(1, 2, 2)
		silhouette_plot(X, preds, preds_clusters, ax=ax2)
		fig.tight_layout(pad=3.0)




# # Finding The Best Number of clusters

def scores_for_all_options(X, true_labels, model_fit, model_predict, num_of_options, options):
	global results

	wcss_scores = np.zeros(num_of_options)  # Within-Cluster-Sum-of-Squares
	aic_scores = np.zeros(num_of_options)
	bic_scores = np.zeros(num_of_options)
	silhouette_scores = np.zeros(num_of_options)
	nmi_scores = np.zeros(num_of_options)

	if type(options) == tuple:
		options_indexes = list(product(np.arange(0, num_of_options[0]), np.arange(0, num_of_options[1])))
	else:
		options_indexes = np.arange(0, num_of_options)

	max_nmi = -np.inf
	max_silhouette = -np.inf
	pbar = tqdm(options_indexes, leave=False, desc="Finding Best Properties")
	for option_index in pbar:
		if type(options) == tuple:
			option = (options[0][option_index[0]], options[1][option_index[1]])
			model = model_fit(*option, X)
		else:
			option = options[option_index]
			model = model_fit(option, X)

		preds = model_predict(model, X)

		# Avoiding large number of clusters (for DBSCAN)
		if isinstance(model, DBSCAN):
			if len(np.unique(preds)) < 3 or len(np.unique(preds)) > 8:
				continue

		if isinstance(model, KMeans):
			wcss_scores[option_index] = model.inertia_
		elif isinstance(model, GaussianMixture):
			aic_scores[option_index] = model.aic(X)
			bic_scores[option_index] = model.bic(X)

		silhouette_scores[option_index] = silhouette_score(X, preds)
		nmi_scores[option_index] = normalized_mutual_info_score(preds, true_labels)

		if nmi_scores[option_index] > max_nmi:
			max_nmi = nmi_scores[option_index]

		if silhouette_scores[option_index] > max_silhouette:
			max_silhouette = silhouette_scores[option_index]

		# pbar.set_description(f"NMI={max_nmi:.4}, silhouette={max_silhouette:.3}")
	# unique_counts = tuple(zip(np.unique(preds, return_counts=True, axis=0)))
	# results = results.append({'eps': option, 'nmi': nmi_scores[option_index], 'silhouette': silhouette_scores[option_index], 'detailed_groups': unique_counts}, ignore_index=True)

	return wcss_scores, aic_scores, bic_scores, silhouette_scores, nmi_scores

# Searching for the best model based on elbow (for kmeans), silhoette and MI scores for one parameter (1d)
def search_for_best_model_1d(X, true_labels, model_fit, model_predict, options, param_name=""):
	# Defining the scores
	scores = scores_for_all_options(X, true_labels, model_fit, model_predict, len(options), options)
	wcss_scores, aic_scores, bic_scores, silhouette_scores, nmi_scores = scores

	if np.any(wcss_scores != 0):
		fig = plt.figure(figsize=(15, 5))
		n_rows, n_cols = 1, 3
		ax = fig.add_subplot(n_rows, n_cols, 3)

		wcss_plot = sns.lineplot(options, wcss_scores, ax=ax, color=color_by_cluster("red"))
		ax.set_xlabel(param_name, fontsize=12)
		ax.set_ylabel("WCSS", fontsize=12)

	elif np.any(aic_scores != 0):
		fig = plt.figure(figsize=(15, 5))
		n_rows, n_cols = 1, 3
		ax = fig.add_subplot(n_rows, n_cols, 3)

		ic_plot = sns.lineplot(options, aic_scores, ax=ax, label="aic", color=color_by_cluster("red"))
		sns.lineplot(options, bic_scores, ax=ax, label="bic", color=color_by_cluster("white"))
		ax.set_xlabel(param_name, fontsize=12)
		ax.set_ylabel("IC", fontsize=12)

		ax.legend(fontsize=12)
	else:
		fig = plt.figure(figsize=(10, 5))
		n_rows, n_cols = 1, 2

	ax = fig.add_subplot(n_rows, n_cols, 2)
	silhouette_plot = sns.lineplot(options, silhouette_scores, ax=ax, color=color_by_cluster("red"))
	ax.set_xlabel(param_name, fontsize=12)
	ax.set_ylabel("Silhouette", fontsize=12)

	ax = fig.add_subplot(n_rows, n_cols, 1)
	mi_plot = sns.lineplot(options, nmi_scores, ax=ax, color=color_by_cluster("red"))
	ax.set_xlabel(param_name, fontsize=12)
	ax.set_ylabel("NMI [bits]", fontsize=12)

	# mi_plot.legend(fontsize = 12)
	fig.tight_layout(pad=2.0)

# Searching for the best model based on elbow (for kmeans), silhoette and MI scores for two parameters (2d)
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

		wcss_plot = ax.plot_surface(X_options, Y_options, wcss_scores, color=color_by_cluster("white"))
		ax.set_xlabel(first_param_name, fontsize=12)
		ax.set_ylabel(second_param_name, fontsize=12)
		ax.set_zlabel("WCSS", fontsize=12)

	elif np.any(aic_scores != 0):
		fig = plt.figure(figsize=(15, 5))
		n_rows, n_cols = 1, 3
		ax = fig.add_subplot(n_rows, n_cols, 3, projection='3d')

		ic_plot = ax.plot_surface(X_options, Y_options, aic_scores, label="aic", color=color_by_cluster("red"))
		ic_plot = ax.plot_surface(X_options, Y_options, bic_scores, label="bic", color=color_by_cluster("white"))
		ax.set_xlabel(first_param_name, fontsize=12)
		ax.set_ylabel(second_param_name, fontsize=12)
		ax.set_zlabel("IC", fontsize=12)
		ax.legend(fontsize=12)
	else:
		fig = plt.figure(figsize=(10, 5))
		n_rows, n_cols = 1, 2

	ax = fig.add_subplot(n_rows, n_cols, 2, projection='3d')
	silhouette_plot = ax.plot_surface(X_options, Y_options, silhouette_scores, color=color_by_cluster("red"),
									  shade=True)
	ax.set_xlabel(first_param_name, fontsize=12)
	ax.set_ylabel(second_param_name, fontsize=12)
	ax.set_zlabel("Silhouette", fontsize=12)

	ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
	mi_plot = ax.plot_surface(X_options, Y_options, nmi_scores, color=color_by_cluster("white"), shade=True)
	ax.set_xlabel(first_param_name, fontsize=12)
	ax.set_ylabel(second_param_name, fontsize=12)
	ax.set_zlabel("NMI [bits]", fontsize=12)

	fig.tight_layout(pad=2.0)





# # Score Methods

# ### P-value

def calculate_pvalue(true_labels, preds, resulted_score, score, score_name, num_of_shuffles, show_graph=False):
	scores_of_shuffles = np.zeros(num_of_shuffles)

	pbar = tqdm(range(num_of_shuffles), leave=False, desc=f"Calculating P-value [{score_name}]")
	for i in pbar:
		new_true_labels = true_labels.copy().replace({"white": 0, "red": 1})
		score_1 = score(pd.DataFrame.sample(new_true_labels, frac=1).reset_index(drop=True), preds)

		new_true_labels = true_labels.copy().replace({"white": 1, "red": 0})
		score_2 = score(pd.DataFrame.sample(new_true_labels, frac=1).reset_index(drop=True), preds)

		scores_of_shuffles[i] = max(score_1, score_2)
	# pbar.set_description(f"{np.max(max(scores_of_shuffles)):.4}")

	# scores_of_shuffles = np.array([score(pd.DataFrame.sample(true_labels, frac=1).reset_index(drop=True), preds) for i in range(num_of_shuffles)])

	if show_graph:
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(1, 1, 1)
		sns.distplot(scores_of_shuffles, kde=True, kde_kws={"color": color_by_cluster("red")},
					 hist_kws={"color": color_by_cluster("red")}, ax=ax)

		# sns.distplot(scores_of_shuffles, ax=ax, color=color_by_cluster("red"))
		ax.set(xlabel=score_name)
	# plt.show()

	return len(np.where(scores_of_shuffles >= resulted_score)[0]) / len(scores_of_shuffles)


def score_with_pvalue(true_labels, preds, score, score_name, num_of_shuffles, match_func=None):
	if match_func is None:
		resulted_score = score(true_labels, preds)
	else:
		tmp_true_labels, tmp_preds, _ = match_labels(true_labels, preds)
		resulted_score = score(tmp_true_labels, tmp_preds)

	if not pvalue_flag:
		return f"{resulted_score:.4}"

	pvalue = calculate_pvalue(true_labels, preds, resulted_score, score, score_name, num_of_shuffles)
	pvalue_resolution = np.log10(num_of_shuffles)

	if pvalue == 0.0:
		return f"{resulted_score:.4}*"
	else:
		return f"{resulted_score:.4} ({pvalue:.pvalue_resolution})"





# ### Clustering Scores

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
			new_preds = np.array(true_labels.copy())
			for label_index in range(len(permutation)):
				np.put(new_preds, np.where(preds == permutation[label_index])[0], unique_true_labels[label_index])

			curr_score = f1_score(true_labels, new_preds, average='macro')
			if curr_score > best_score:
				best_score = curr_score
				best_preds = new_preds
				best_matching = permutation

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

	get_score_func = lambda score, score_name, match_func=None: score_with_pvalue(true_labels, preds, score, score_name,
																				  pvalue_num_of_shuffles, match_func)

	mi_score = mutual_info_score(true_labels, preds)
	nmi_score = get_score_func(normalized_mutual_info_score, "NMI")

	print(f"Silhouette score: Best={max(silhouette_scores):.4}, Wrost={min(silhouette_scores):.4}, Total={silhouette_total_score:.4}")
	print(f"MI scores: Standard={mi_score:.4}, Normalized={nmi_score}")

	tmp_true_labels, tmp_preds, matching = match_labels(true_labels, preds)
	# lambda true_labels, preds: np.sum([1 for i in range(len(preds)) if true_labels[i] == preds[i]]) / len(preds)
	acc = get_score_func(accuracy_score, "ACC", match_func=match_labels)
	recall = recall_score(tmp_true_labels, tmp_preds, average='macro')
	precision = precision_score(tmp_true_labels, tmp_preds, average='macro')
	f1 = get_score_func(lambda true_labels, preds: f1_score(true_labels, preds, average='macro'), "F1",
						match_func=match_labels)

	print(np.take(np.unique(tmp_true_labels), matching))
	confusion_mat = pd.DataFrame(
		confusion_matrix(tmp_true_labels, tmp_preds, labels=np.take(np.unique(tmp_true_labels), matching)))
	# confusion_mat.rename(columns=np.unique(tmp_true_labels) index={0:matching[0], 1:matching[1]}, inplace=True)
	# confusion_mat.rename(columns=np.unique(tmp_true_labels), inplace=True)
	confusion_mat.columns = np.take(np.unique(tmp_true_labels), matching)

	print(f"Precision: {precision:.4}, Recall: {recall:.4}, f1-score: {f1}, Accuracy: {acc}")
	print(confusion_mat)

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