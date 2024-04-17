import numpy as np
from torchvision import datasets, transforms
from src.Utilities.data.CelebaDataset import CelebaDataset


def get_images_by_label(dataset, num_images=50):
    targets = np.unique(dataset.targets)
    images_by_label = {label: [] for label in targets}
    for i in range(min(num_images, len(dataset))):
        images_by_label[dataset[i][1]].append(dataset[i][0][0].numpy())

    for label in targets:
        images_by_label[label] = np.concatenate(images_by_label[label])
    return images_by_label


def get_mnist(target_label):
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
