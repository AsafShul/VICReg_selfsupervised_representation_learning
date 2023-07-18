"""
a main file that loads your VICReg model,
and performs Anomaly Detection using its representations as in Sec. 4,
Anomaly Detection Q1, Then plots the ROC Curve
"""


import os
import torch
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from VICRegModel import VICRegModel
from CIFAR10Dataset import DataCreator
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors


if __name__ == '__main__':
    # load model:
    base_model = VICRegModel()
    base_model.load_model()

    # get embeddings:
    if os.path.isfile(os.path.join(utils.get_res_path(), 'loaders', 'mnist_embedding_train_loader.pt')):
        mnist_embedding_train_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', 'mnist_embedding_train_loader.pt'))
        mnist_embedding_test_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', 'mnist_embedding_test_loader.pt'))
    else:
        mnist_embedding_train_loader, mnist_embedding_test_loader = \
            DataCreator.get_CIFAR10MNIST_loaders(base_model=base_model, neighbors_model=None)

    vicreg_train_embeddings = mnist_embedding_train_loader.dataset.base_embeddings
    vicreg_test_embeddings = mnist_embedding_test_loader.dataset.base_embeddings

    vicreg_knn = NearestNeighbors(n_neighbors=2).fit(vicreg_train_embeddings)

    density_vicreg = [utils.calc_inv_density_score(x, vicreg_knn) for x in tqdm(vicreg_test_embeddings)]

    # plot roc curve:
    test_labels = mnist_embedding_test_loader.dataset.targets

    vicreg_fpr, vicreg_tpr, vicreg_thresholds = roc_curve(test_labels, density_vicreg)

    plt.plot(vicreg_fpr, vicreg_tpr, label='VICReg')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.text(0.5, 0.5, f'VICReg AUC: {auc(vicreg_fpr, vicreg_tpr)}',
             horizontalalignment='center', verticalalignment='center')
    plt.legend()
    plt.show()

    vicreg_anomalous_idx = np.argsort(density_vicreg)[-7:]
    vicreg_anomalous_images = [mnist_embedding_test_loader.dataset.get_raw(i) for i in vicreg_anomalous_idx]

    # plot images:
    fig, axes = plt.subplots(1, 7)
    for i, ax in enumerate(axes):
        ax.imshow(vicreg_anomalous_images[i])
        ax.set_title(f'{i}')
        ax.axis('off')
    plt.savefig(os.path.join(utils.get_res_path(), 'plots', 'vicreg_anomalous_images.png'))
    plt.show()
