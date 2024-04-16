from torchvision import datasets, transforms
from src.Utilities.data.CelebaDataset import CelebaDataset


def get_images_by_label(dataset, label, num_images=15):
    label_indices = [i for i in range(len(dataset)) if dataset.targets[i] == label]
    images = []

    for i in range(min(num_images, len(label_indices))):
        index = label_indices[i]
        image, _ = dataset[index]
        images.append(image[0].numpy())

    return images


def get_mnist(target_label):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the MNIST dataset
    train_dataset = datasets.MNIST(root="~/GITHUB/DATASETS", train=True, download=True, transform=transform)

    # Retrieve images by label
    return get_images_by_label(train_dataset, target_label)


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
