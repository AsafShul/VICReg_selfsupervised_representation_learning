import os
import torch

import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from VICRegModel import VICRegModel
from CIFAR10Dataset import DataCreator
from LinearProbModel import LinearProbModel

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc, silhouette_score

REDUCTION_DIM = 2


class Questions:
    @staticmethod
    def p1q1_training(train_loader, test_loader, train_model):
        print('Q1: Training')
        model = VICRegModel()
        model.fit(train_loader, test_loader) if train_model else model.load_model()
        return model

    @staticmethod
    def p1q2_pca_tsne_plot(embeddings_loader, postfix='', reduction_dim=REDUCTION_DIM):
        print('Q2: Plotting PCA and TSNE')
        train_dataset = embeddings_loader.dataset.embeddings
        labels = embeddings_loader.dataset.targets
        print('\tfitting pca... ', end='')
        pca_embeddings = PCA(n_components=reduction_dim).fit_transform(train_dataset)
        print('done!')
        print('\tfitting tsne... ', end='')
        tsne_embeddings = TSNE(n_components=reduction_dim, verbose=1).fit_transform(train_dataset)
        print('done!')

        print('plotting...')
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title('PCA' + postfix)
        scatter_pca = plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=labels, cmap='tab10', s=1)
        plt.subplot(1, 2, 2)
        plt.title('TSNE' + postfix)
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap='tab10', s=1)

        classes = embeddings_loader.dataset.dataset.classes
        plt.legend(scatter_pca.legend_elements()[0], classes,
                   loc='lower center', bbox_to_anchor=(-0.1, -0.15),
                   ncol=len(classes))

        plt.savefig(os.path.join(utils.get_res_path(), 'plots', 'lin_prob' + postfix + '.png'))
        plt.show()

    @staticmethod
    def p1q3_lin_prob(train_loader, test_loader):
        print('Q3: Linear Probing')
        input_dim = train_loader.dataset.embeddings.shape[1]
        output_dim = len(train_loader.dataset.dataset.classes)

        lin = LinearProbModel(input_dim, output_dim)
        lin.fit(train_loader)
        lin.evaluate(test_loader)

    @staticmethod
    def p1q4_0varloss(train_loader, test_loader, train_model):
        print('Q4: Ablation 1 - No Variance Term')
        model = VICRegModel(w_var=0)
        model.MODEL_NAME = model.MODEL_NAME.replace('.', '_0var.')
        model.fit(train_loader, test_loader) if train_model else model.load_model()

        force_create = False
        name = 'ZeroVar_embedding'
        if os.path.isfile(os.path.join(utils.get_res_path(), 'loaders', f'{name}_train_loader.pt')) and not force_create:
            embedding_train_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', f'{name}_train_loader.pt'))
            embedding_test_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', f'{name}_test_loader.pt'))
        else:
            embedding_train_loader, embedding_test_loader = DataCreator.get_embedded_CIFAR10_loaders(model)
            torch.save(embedding_train_loader, os.path.join(utils.get_res_path(), 'loaders', f'{name}_train_loader.pt'))
            torch.save(embedding_test_loader, os.path.join(utils.get_res_path(), 'loaders', f'{name}_test_loader.pt'))

        Questions.p1q2_pca_tsne_plot(embedding_train_loader, ' - 0 variance term (μ = 0)')
        Questions.p1q3_lin_prob(embedding_train_loader, embedding_test_loader)
        return model

    @staticmethod
    def p1q5():
        pass

    @staticmethod
    def p1q6_neighbors_views(train_loader, test_loader, train_model):
        print('Q6: Ablation 2 - Neighbors Views')
        model = VICRegModel()
        model.MODEL_NAME = model.MODEL_NAME.replace('.', '_neighbors.')
        model.fit(train_loader, test_loader, epoch_num=1) if train_model else model.load_model()
        embedding_train_loader, embedding_test_loader = DataCreator.get_embedded_CIFAR10_loaders(model)
        Questions.p1q3_lin_prob(embedding_train_loader, embedding_test_loader)
        return model, embedding_train_loader, embedding_test_loader

    @staticmethod
    def p1q8_compare(neighbors_embeddings_loader, vicreg_embeddings_loader):
        sample_neighbors_data, sample_neighbors_embeddings, sample_neighbors_classes = \
            DataCreator.get_sample_per_class(neighbors_embeddings_loader)

        sample_vicreg_data, sample_vicreg_embeddings, sample_vicreg_classes = \
            DataCreator.get_sample_per_class(vicreg_embeddings_loader)

        neighbors_embeddings = neighbors_embeddings_loader.dataset.embeddings
        neighbors_images = neighbors_embeddings_loader.dataset.dataset.data

        vicreg_embeddings = vicreg_embeddings_loader.dataset.embeddings
        vicreg_images = vicreg_embeddings_loader.dataset.dataset.data

        names = neighbors_embeddings_loader.dataset.dataset.classes

        # nearest images:
        neighbors_knn = NearestNeighbors(n_neighbors=6).fit(neighbors_embeddings)
        vicreg_knn = NearestNeighbors(n_neighbors=6).fit(vicreg_embeddings)

        Questions.q8_plot_helper('vicreg', names, sample_vicreg_classes,
                                 sample_vicreg_embeddings, vicreg_images, vicreg_knn)

        Questions.q8_plot_helper('neighbors', names, sample_neighbors_classes,
                                 sample_neighbors_embeddings, neighbors_images, neighbors_knn)

        # most distant images:
        far_neighbors_knn = NearestNeighbors(n_neighbors=len(neighbors_embeddings)).fit(neighbors_embeddings)
        far_vicreg_knn = NearestNeighbors(n_neighbors=len(neighbors_embeddings)).fit(vicreg_embeddings)

        Questions.q8_plot_helper('vicreg', names, sample_vicreg_classes,
                                 sample_vicreg_embeddings, vicreg_images, far_vicreg_knn, far=True)

        Questions.q8_plot_helper('neighbors', names, sample_neighbors_classes,
                                 sample_neighbors_embeddings, neighbors_images, far_neighbors_knn, far=True)

    @staticmethod
    def q8_plot_helper(model_name, names, sample_classes, sample_embeddings, images, knn, far=False):
        for c, sample in zip(sample_classes, sample_embeddings):
            if not far:
                vicreg_idx = knn.kneighbors(sample.reshape(1, -1), return_distance=False)[0]
            else:
                vicreg_idx_all = knn.kneighbors(sample.reshape(1, -1), return_distance=False)[0]
                vicreg_idx = np.array([vicreg_idx_all[0]] + list(vicreg_idx_all[-5:]))
            neighbors_images = images[vicreg_idx]
            c_name = names[c]

            # plot images:
            fig, axes = plt.subplots(1, 6)
            for i, ax in enumerate(axes):
                ax.imshow(neighbors_images[i])
                ax.set_title([f'original[{c_name}]', '1', '2', '3', '4', '5'][i])
                ax.axis('off')
            plt.savefig(os.path.join(utils.get_res_path(), 'plots', f'{model_name}_{c_name}{"_far" if far else ""}.png'))
            plt.show()

    @staticmethod
    def p2(mnist_embedding_train_loader, mnist_embedding_test_loader):
        vicreg_train_embeddings = mnist_embedding_train_loader.dataset.base_embeddings
        neighbors_train_embeddings = mnist_embedding_train_loader.dataset.neighbors_embeddings

        vicreg_test_embeddings = mnist_embedding_test_loader.dataset.base_embeddings
        neighbors_test_embeddings = mnist_embedding_test_loader.dataset.neighbors_embeddings

        vicreg_knn = NearestNeighbors(n_neighbors=2).fit(vicreg_train_embeddings)
        neighbors_knn = NearestNeighbors(n_neighbors=2).fit(neighbors_train_embeddings)

        print('calculating inv density for vicreg')
        density_vicreg = [utils.calc_inv_density_score(x, vicreg_knn) for x in tqdm(vicreg_test_embeddings)]
        print('calculating inv density for neighbors')
        density_neighbors = [utils.calc_inv_density_score(x, neighbors_knn) for x in tqdm(neighbors_test_embeddings)]

        print(f'mean of the inv density score over the test set for VICReg: {np.mean(density_vicreg)}')
        print()
        print(f'mean of the inv density score over the test set for VICReg + Neighbors: {np.mean(density_neighbors)}')

        # plot roc curve:
        test_labels = mnist_embedding_test_loader.dataset.targets

        vicreg_fpr, vicreg_tpr, _ = roc_curve(test_labels, density_vicreg)
        neighbors_fpr, neighbors_tpr, _ = roc_curve(test_labels, density_neighbors)

        plt.plot(vicreg_fpr, vicreg_tpr, label='VICReg')
        plt.plot(neighbors_fpr, neighbors_tpr, label='VICReg + Neighbors')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.text(0.5, 0.5, f'VICReg AUC: {auc(vicreg_fpr, vicreg_tpr)}\nVICReg + Neighbors AUC: '
                           f'{auc(neighbors_fpr, neighbors_tpr)}',
                 horizontalalignment='center', verticalalignment='center')
        plt.legend()
        plt.savefig(os.path.join(utils.get_res_path(), 'plots', 'roc_curve.png'))
        plt.show()

        vicreg_anomalous_idx = np.argsort(density_vicreg)[-7:]
        neighbors_anomalous_idx = np.argsort(density_neighbors)[-7:]

        vicreg_anomalous_images = [mnist_embedding_test_loader.dataset.get_raw(i) for i in vicreg_anomalous_idx]
        neighbors_anomalous_images = [mnist_embedding_test_loader.dataset.get_raw(i) for i in neighbors_anomalous_idx]

        # plot images:
        fig, axes = plt.subplots(1, 7)
        for i, ax in enumerate(axes):
            ax.imshow(vicreg_anomalous_images[i])
            ax.set_title(f'{i}')
            ax.axis('off')
        plt.savefig(os.path.join(utils.get_res_path(), 'plots', 'vicreg_anomalous_images.png'))
        plt.show()

        fig, axes = plt.subplots(1, 7)
        for i, ax in enumerate(axes):
            ax.imshow(neighbors_anomalous_images[i])
            ax.set_title(f'{i}')
            ax.axis('off')
        plt.savefig(os.path.join(utils.get_res_path(), 'plots', 'neighbors_anomalous_images.png'))
        plt.show()

        # clustering:
        vicreg_kmeans = KMeans(n_clusters=10).fit(vicreg_train_embeddings)
        neighbors_kmeans = KMeans(n_clusters=10).fit(neighbors_train_embeddings)

        # plot the clusters, with tsne and pca:
        vicreg_tsne = TSNE(n_components=2).fit_transform(vicreg_train_embeddings)
        neighbors_tsne = TSNE(n_components=2).fit_transform(neighbors_train_embeddings)

        vicreg_centers = np.stack([vicreg_tsne[vicreg_kmeans.labels_ == i].mean(axis=0) for i in range(10)], axis=0)
        neighbors_centers = np.stack([neighbors_tsne[neighbors_kmeans.labels_ == i].mean(axis=0) for i in range(10)], axis=0)

        fig, axes = plt.subplots(1, 2)
        axes[0].scatter(vicreg_tsne[:, 0], vicreg_tsne[:, 1], c=vicreg_kmeans.labels_, s=1)
        axes[0].scatter(vicreg_centers[:, 0], vicreg_centers[:, 1], c='black', s=50)
        axes[0].set_title('VICReg model clustering with TSNE - colored by cluster')
        axes[1].scatter(vicreg_tsne[:, 0], vicreg_tsne[:, 1], c=mnist_embedding_train_loader.dataset.cifar10.targets, s=1)
        axes[1].scatter(vicreg_centers[:, 0], vicreg_centers[:, 1], c='black', s=50)
        axes[1].set_title('VICReg model clustering with TSNE - colored by class')
        fig.set_size_inches(14, 7)
        plt.show()

        fig, axes = plt.subplots(1, 2)
        axes[0].scatter(neighbors_tsne[:, 0], neighbors_tsne[:, 1], c=neighbors_kmeans.labels_, s=1)
        axes[0].scatter(neighbors_centers[:, 0], neighbors_centers[:, 1], c='black', s=50)
        axes[0].set_title('VICReg + Neighbors model clustering with TSNE - colored by cluster')
        axes[1].scatter(neighbors_tsne[:, 0], neighbors_tsne[:, 1], c=mnist_embedding_train_loader.dataset.cifar10.targets, s=1)
        axes[1].scatter(neighbors_centers[:, 0], neighbors_centers[:, 1], c='black', s=50)
        axes[1].set_title('VICReg + Neighbors model clustering with TSNE - colored by class')
        fig.set_size_inches(14, 7)
        plt.show()

        # clac silhouette score:
        vicreg_silhouette_score = silhouette_score(vicreg_train_embeddings, vicreg_kmeans.labels_)
        neighbors_silhouette_score = silhouette_score(neighbors_train_embeddings, neighbors_kmeans.labels_)

        print(f'VICReg silhouette score: {vicreg_silhouette_score}')
        print(f'VICReg + Neighbors silhouette score: {neighbors_silhouette_score}')
