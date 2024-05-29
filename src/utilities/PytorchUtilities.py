import numpy as np
from torchvision import datasets, transforms
from src.utilities.data.CelebaDataset import CelebaDataset
from sklearn.datasets import load_svmlight_file, make_friedman1


def get_images_by_label(dataset, num_images=6000):
    targets = np.unique(dataset.targets)
    images_by_label = {label: [] for label in targets}
    for i in range(min(num_images, len(dataset))):
        images_by_label[dataset[i][1]].append(dataset[i][0][0].view(-1).numpy())

    for label in targets:
        images_by_label[label] = np.array(images_by_label[label])
    return list(images_by_label.values())


def get_mnist():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the MNIST dataset
    train_dataset = datasets.MNIST(root="~/GITHUB/DATASETS", train=True, download=True, transform=transform)

    # Retrieve images by label
    return get_images_by_label(train_dataset)


def get_celeba(nb_clients):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.CenterCrop(148), transforms.Resize(32),
                                    ])

    trainset = CelebaDataset(data_dir="~/GITHUB/DATASETS/celeba/img_align_celeba",
                             partition_file_path='~/GITHUB/DATASETS/celeba/list_eval_partition.csv',
                             identity_file_path='~/GITHUB/DATASETS/celeba/identity_CelebA.txt',
                            split=[0, 1],
                            transform=transform)

    return [trainset.__getitem_by_identity__(i) for i in range(1, nb_clients+1)]


def get_friedman(nb_clients):
    dataset = []
    for i in nb_clients:
        dataset.append(make_friedman1(n_samples = 100, n_features=1000, random_state=i, noise=i)[0])
    return dataset


def get_w8a(nb_clients):
    raw_X, raw_Y = load_svmlight_file("../../DATASETS/w8a/w8a")
    raw_X = raw_X.todense()
    return np.array_split(np.array(raw_X), nb_clients)


def get_real_sim(nb_clients):
    print("Get the real-sim dataset.")
    raw_X, raw_Y = load_svmlight_file("../../DATASETS/real-sim/real-sim.bz2")
    print("Densification.")
    raw_X = raw_X.todense()
    return np.array_split(np.array(raw_X), nb_clients)