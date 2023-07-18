import os
import torch
import utils
from Questions import Questions
from VICRegModel import VICRegModel
from CIFAR10Dataset import DataCreator

TRAIN_MODEL_BASE = False
TRAIN_MODEL_ZERO_VAR = False
TRAIN_MODEL_NEIGHBORS_VIEWS = False

if __name__ == '__main__':
    # --------------------------- q1 Training --------------------------------
    base_train_loader, base_test_loader = DataCreator.get_base_CIFAR10_loaders()
    views_train_loader, views_test_loader = DataCreator.get_views_CIFAR10_loaders()

    base_model = Questions.p1q1_training(views_train_loader, base_test_loader, train_model=TRAIN_MODEL_BASE)

    # ----------------------------- q2 PCA and t-SNE --------------------------------

    if os.path.isfile(os.path.join(utils.get_res_path(), 'loaders',  'embedding_train_loader.pt')):
        embedding_train_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', 'embedding_train_loader.pt'))
        embedding_test_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', 'embedding_test_loader.pt'))
    else:
        embedding_train_loader, embedding_test_loader = DataCreator.get_embedded_CIFAR10_loaders(base_model)
        torch.save(embedding_train_loader, os.path.join(utils.get_res_path(), 'loaders', 'embedding_train_loader.pt'))
        torch.save(embedding_test_loader, os.path.join(utils.get_res_path(), 'loaders', 'embedding_test_loader.pt'))

    Questions.p1q2_pca_tsne_plot(embedding_train_loader)

    # ----------------------------- q3 Linear Probing --------------------------------
    Questions.p1q3_lin_prob(embedding_train_loader, embedding_test_loader)

    # ----------------------------- q4 Ablation 1 - No Variance Term --------------------------------
    zero_var_model = Questions.p1q4_0varloss(views_train_loader, views_test_loader, train_model=TRAIN_MODEL_ZERO_VAR)

    # ----------------------------- Ablation 2 - No Amortization --------------------------------
    Questions.p1q5()
    # ----------------------- Ablation 3 - No Generated Neighbors --------------------------------

    if os.path.isfile(os.path.join(utils.get_res_path(), 'loaders', 'neighbors_embedding_train_loader.pt')):
        print('Loading neighbors loaders...')
        no_generated_neighbors_model = VICRegModel()
        no_generated_neighbors_model.MODEL_NAME = no_generated_neighbors_model.MODEL_NAME.replace('.', '_neighbors.')
        no_generated_neighbors_model.load_model()

        neighbors_embedding_train_loader = torch.load(
            os.path.join(utils.get_res_path(), 'loaders', 'neighbors_embedding_train_loader.pt'))
        neighbors_embedding_test_loader = torch.load(
            os.path.join(utils.get_res_path(), 'loaders', 'neighbors_embedding_test_loader.pt'))
    else:
        neighbors_train_loader, neighbors_test_loader = \
            DataCreator.get_neighbors_CIFAR10_loaders(embedding_train_loader)

        no_generated_neighbors_model, neighbors_embedding_train_loader, neighbors_embedding_test_loader = \
            Questions.p1q6_neighbors_views(neighbors_train_loader, neighbors_test_loader,
                                           train_model=TRAIN_MODEL_NEIGHBORS_VIEWS)

        torch.save(neighbors_embedding_train_loader,
                   os.path.join(utils.get_res_path(), 'loaders', 'neighbors_embedding_train_loader.pt'))

        torch.save(neighbors_embedding_test_loader,
                   os.path.join(utils.get_res_path(), 'loaders', 'neighbors_embedding_test_loader.pt'))

    # ---------------------------- q8 Retrieval Evaluation --------------------------------
    Questions.p1q8_compare(neighbors_embedding_train_loader, embedding_train_loader)

    # ---------------------------- part 2 MNIST anomalies --------------------------------

    if os.path.isfile(os.path.join(utils.get_res_path(), 'loaders', 'mnist_embedding_train_loader.pt')):
        print('Loading MNIST loaders...')
        mnist_embedding_train_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', 'mnist_embedding_train_loader.pt'))
        mnist_embedding_test_loader = torch.load(os.path.join(utils.get_res_path(), 'loaders', 'mnist_embedding_test_loader.pt'))
    else:
        print('Creating MNIST loaders...')
        mnist_embedding_train_loader, mnist_embedding_test_loader = \
            DataCreator.get_CIFAR10MNIST_loaders(base_model, no_generated_neighbors_model)

        torch.save(mnist_embedding_train_loader, os.path.join(utils.get_res_path(), 'loaders', 'mnist_embedding_train_loader.pt'))
        torch.save(mnist_embedding_test_loader, os.path.join(utils.get_res_path(), 'loaders', 'mnist_embedding_test_loader.pt'))

    Questions.p2(mnist_embedding_train_loader, mnist_embedding_test_loader)
