"""
prepare a main file that loads your submitted weights
for your (original) VICReg implementation, performs linear probing on it (Sec. 3, Q3),
and plots the closest retrievals for a sample from each class (Sec. 3, Q8).
"""

import os
import torch
import utils
from Questions import Questions
from VICRegModel import VICRegModel
from CIFAR10Dataset import DataCreator
from sklearn.neighbors import NearestNeighbors

if __name__ == '__main__':
    # load model:
    model = VICRegModel()
    model.load_model()

    if os.path.isfile(os.path.join(utils.get_res_path(), 'loaders', 'embedding_train_loader.pt')):
        embedding_train_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', 'embedding_train_loader.pt'))
        embedding_test_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', 'embedding_test_loader.pt'))
    else:
        embedding_train_loader, embedding_test_loader = DataCreator.get_embedded_CIFAR10_loaders(model)

    # linear probing:
    Questions.p1q3_lin_prob(embedding_train_loader, embedding_test_loader)

    # plot closest retrievals:
    sample_vicreg_data, sample_vicreg_embeddings, sample_vicreg_classes = \
        DataCreator.get_sample_per_class(embedding_train_loader)

    vicreg_embeddings = embedding_train_loader.dataset.embeddings
    vicreg_images = embedding_train_loader.dataset.dataset.data

    vicreg_knn = NearestNeighbors(n_neighbors=6).fit(vicreg_embeddings)

    names = embedding_train_loader.dataset.dataset.classes

    Questions.q8_plot_helper('vicreg', names, sample_vicreg_classes, sample_vicreg_embeddings,
                             vicreg_images, vicreg_knn)
