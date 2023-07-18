import torch
import utils
import torchvision
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from VICRegModel import VICRegModel
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from augmentations import train_transform, test_transform

NEIGHBORS = 3
NUM_WORKERS = 0
BATCH_SIZE = 256

ROOT = './data'
TORCH_TRANSFORM = transforms.Compose([transforms.ToTensor()])
BASE_TRANSFORM = transforms.Compose([transforms.ToTensor(), test_transform])
MNIST_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])


class DataCreator:
    @staticmethod
    def get_base_CIFAR10_loaders():
        train_loader = DataCreator.get_base_CIFAR10_loader(train=True)
        test_loader = DataCreator.get_base_CIFAR10_loader(train=False)
        return train_loader, test_loader

    @staticmethod
    def get_views_CIFAR10_loaders():
        train_loader = DataCreator.get_views_CIFAR10_loader(train=True)
        test_loader = DataCreator.get_views_CIFAR10_loader(train=False)
        return train_loader, test_loader

    @staticmethod
    def get_embedded_CIFAR10_loaders(model):
        train_loader = DataCreator.get_embedded_CIFAR10_loader(model, train=True)
        test_loader = DataCreator.get_embedded_CIFAR10_loader(model, train=False)
        return train_loader, test_loader

    @staticmethod
    def get_neighbors_CIFAR10_loaders(embedded_train_loader=None, model=None, n_neighbors=NEIGHBORS + 1):
        train_loader = DataCreator.get_neighbors_CIFAR10_loader(embedded_train_loader=embedded_train_loader,
                                                                model=model, train=True, n_neighbors=n_neighbors)
        test_loader = DataCreator.get_neighbors_CIFAR10_loader(embedded_train_loader=embedded_train_loader,
                                                               model=model, train=False, n_neighbors=n_neighbors)
        return train_loader, test_loader

    @staticmethod
    def get_CIFAR10MNIST_loaders(base_model, neighbors_model):
        train_loader = DataCreator.get_CIFAR10MNIST_loader(base_model, neighbors_model, train=True)
        test_loader = DataCreator.get_CIFAR10MNIST_loader(base_model, neighbors_model, train=False)
        return train_loader, test_loader

    # -------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_sample_per_class(embeddings_loader):
        train_set = embeddings_loader.dataset.dataset
        embeddings = embeddings_loader.dataset.embeddings
        samples = {}
        for i in range(len(train_set)):
            if len(samples) == len(train_set.classes):
                break

            x, y = train_set[i]
            e = embeddings[i]
            if y in samples.keys():
                continue

            samples[y] = x, e

        images = [sample[0] for sample in samples.values()]
        embeddings = [sample[1] for sample in samples.values()]
        classes = list(samples.keys())

        return images, embeddings, classes

    @staticmethod
    def get_base_CIFAR10_loader(train=True, root=ROOT, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS):
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=BASE_TRANSFORM)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader

    @staticmethod
    def get_views_CIFAR10_loader(train=True, root=ROOT, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS):
        loader = torch.utils.data.DataLoader(CIFAR10ViewsDataset(train=train, root=root),
                                             batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader

    @staticmethod
    def get_embedded_CIFAR10_loader(model, train=True, root=ROOT, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS):
        loader = torch.utils.data.DataLoader(EmbeddingsCIFAR10Dataset(model, train=train, root=root),
                                             batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader

    @staticmethod
    def get_neighbors_CIFAR10_loader(embedded_train_loader=None, model=None, train=True, root=ROOT,
                                     batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                     n_neighbors=NEIGHBORS + 1):
        loader = torch.utils.data.DataLoader(CIFAR10NeighborsDataset(loader=embedded_train_loader,
                                                                     model=model, train=train, root=root,
                                                                     batch_size=batch_size, shuffle=shuffle,
                                                                     num_workers=num_workers, n_neighbors=n_neighbors),
                                             batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader

    @staticmethod
    def get_CIFAR10MNIST_loader(base_model, neighbors_model, train=True, root=ROOT, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS):
        loader = torch.utils.data.DataLoader(CIFAR10MNISTDataset(base_model, neighbors_model, train=train, root=root),
                                             batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader


class CIFAR10ViewsDataset(Dataset):
    """view1, view2, classes"""
    VIEW1_IDX = 0
    VIEW2_IDX = 1
    CLASSES_IDX = 2

    def __init__(self, train=True, root=ROOT):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=TORCH_TRANSFORM)
        self.device = utils.get_device()

    def __getitem__(self, index):
        return (train_transform(self.dataset[index][0].to(self.device)),
                train_transform(self.dataset[index][0].to(self.device)),
                torch.tensor(self.dataset[index][1]).to(self.device))

    def __len__(self):
        return len(self.dataset)


class EmbeddingsCIFAR10Dataset(Dataset):
    """embeddings, classes, images"""
    EMBEDDINGS_IDX = 0
    CLASSES_IDX = 1
    IMAGES_IDX = 2

    def __init__(self, model: VICRegModel, train=True, root=ROOT):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=BASE_TRANSFORM)
        self.device = utils.get_device()

        model.eval()
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=NUM_WORKERS)

            self.embeddings = torch.vstack([model(image) for image in tqdm(loader)]).cpu().numpy()
            self.data = torch.vstack([image[0] for image in tqdm(loader)]).cpu().numpy()
            self.targets = torch.hstack([image[1] for image in tqdm(loader)]).cpu().numpy()

    def __getitem__(self, index):
        e = torch.tensor(self.embeddings[index]).to(self.device)
        c = torch.tensor(self.targets[index]).to(self.device)
        x = torch.tensor(self.data[index]).to(self.device)

        return e, c, x

    def __len__(self):
        return len(self.dataset)


class CIFAR10NeighborsDataset(Dataset):
    """images, neighbors, classes"""
    IMAGE_VIEW_IDX = 0
    NEIGHBOR_VIEW_IDX = 1
    CLASSES_IDX = 2

    def __init__(self, loader=None, model=None, train=True, root=ROOT,
                 batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, n_neighbors=NEIGHBORS + 1):
        assert loader is not None or model is not None, 'Must provide either embedded or model'
        self.n_neighbors = n_neighbors
        self.device = utils.get_device()

        if loader is None:
            model.eval()
            with torch.no_grad():
                loader = DataCreator.get_embedded_CIFAR10_loader(model, train=train, root=root,
                                                                 batch_size=batch_size, shuffle=shuffle,
                                                                 num_workers=num_workers)
        self.data = torch.vstack([b[loader.dataset.IMAGES_IDX] for b in loader]).cpu().numpy()
        self.targets = torch.hstack([b[loader.dataset.CLASSES_IDX] for b in loader]).cpu().numpy()
        self.embeddings = torch.vstack([b[loader.dataset.EMBEDDINGS_IDX] for b in loader]).cpu().numpy()

        self.knn = NearestNeighbors(n_neighbors=n_neighbors).fit(self.embeddings)

    def __getitem__(self, index):

        possible_neighbors = self.knn.kneighbors(self.embeddings[index].reshape(1, -1), return_distance=False)[0]
        neighbor_idx = np.random.randint(self.n_neighbors)

        img = torch.tensor(self.data[index]).to(self.device)
        neighbor = torch.tensor(self.data[possible_neighbors[neighbor_idx]]).to(self.device)
        target = torch.tensor(self.targets[index]).to(self.device)

        return img, neighbor, target

    def __len__(self):
        return len(self.data)


class CIFAR10MNISTDataset(Dataset):
    def __init__(self, base_model, neighbors_model=None, train=True, root=ROOT):
        assert base_model is not None, 'Must provide model'
        self.train = train
        self.device = utils.get_device()

        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.mnist = torchvision.datasets.MNIST(root=root, train=train, download=True)

        with torch.no_grad():
            base_model.eval()
            self.base_cifar10_embeddings = torch.vstack([base_model(BASE_TRANSFORM(img).unsqueeze(0)) for img, _ in tqdm(self.cifar10)]).cpu().numpy()
            self.base_mnist_embeddings = torch.vstack([base_model(MNIST_TRANSFORM(img).unsqueeze(0)) for img, _ in tqdm(self.mnist)]).cpu().numpy()

            if neighbors_model is None:
                self.neighbors_cifar10_embeddings = self.base_cifar10_embeddings
                self.neighbors_mnist_embeddings = self.base_mnist_embeddings
            else:
                neighbors_model.eval()
                self.neighbors_cifar10_embeddings = torch.vstack([neighbors_model(BASE_TRANSFORM(img).unsqueeze(0)) for img, _ in tqdm(self.cifar10)]).cpu().numpy()
                self.neighbors_mnist_embeddings = torch.vstack([neighbors_model(MNIST_TRANSFORM(img).unsqueeze(0)) for img, _ in tqdm(self.mnist)]).cpu().numpy()

        self.cifar10_targets = np.zeros(len(self.cifar10))
        self.mnist_targets = np.ones(len(self.mnist))

        if train:
            self.base_embeddings = self.base_cifar10_embeddings
            self.neighbors_embeddings = self.neighbors_cifar10_embeddings
            self.targets = self.cifar10_targets

        else:
            self.base_embeddings = np.vstack([self.base_cifar10_embeddings, self.base_mnist_embeddings])
            self.neighbors_embeddings = np.vstack([self.neighbors_cifar10_embeddings, self.neighbors_mnist_embeddings])
            self.targets = np.hstack([self.cifar10_targets, self.mnist_targets])

    def __getitem__(self, index):
        assert ((index < len(self.cifar10) + len(self.mnist)) and not self.train) or \
               (self.train and index < len(self.cifar10)), 'Index out of range'
        if index < len(self.cifar10):
            base_embeddings = self.base_cifar10_embeddings
            neighbors_embeddings = self.neighbors_cifar10_embeddings
            targets = self.cifar10_targets
            idx = index
        else:
            base_embeddings = self.base_mnist_embeddings
            neighbors_embeddings = self.neighbors_mnist_embeddings
            targets = self.mnist_targets
            idx = index - len(self.cifar10)

        img = torch.tensor(base_embeddings[idx])
        neighbor = torch.tensor(neighbors_embeddings[idx])
        target = torch.tensor(targets[idx])

        return img, neighbor, target

    def __len__(self):
        return len(self.cifar10) + len(self.mnist)

    def get_raw(self, index):
        if index < len(self.cifar10):
            return self.cifar10[index][0]
        else:
            return self.mnist[index - len(self.cifar10)][0]
