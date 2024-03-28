import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset
from .randaugment import RandAugmentMC
from PIL import Image

def load_svhn(partial_rate, root="./data"):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    temp_train = datasets.SVHN(root=root, split='train', download=True)
    # print(temp_train.target)
    data, labels = temp_train.data, torch.Tensor(temp_train.labels).long()
    # get original data and labels

    test_dataset = datasets.SVHN(root, transform=test_transform, split='test', download=True)
    # set test dataloader
    
    partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    # generate partial labels
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = SVHN_Augmentention(data, partialY.float(), labels.float(),transform=TransformFixMatch(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5)))

    return partial_matrix_dataset,test_dataset

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong1 = self.strong(x)
        strong2 = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong1), self.normalize(strong2)
        else:
            return weak, strong

class SVHN_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels,transform):
        self.transform=transform
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.init_index()
        self.return_truelabel=False

    def set_index(self, indexes=None,reture_truelabel=False,selected_labels=None):
        self.return_truelabel=reture_truelabel
        self.true_labels_index=self.true_labels[indexes]
        self.images_index = self.images[indexes]
        self.given_label_matrix_index=self.given_label_matrix[indexes]
        if selected_labels is not None:
            self.selected_label=selected_labels
        # print(selected_label)
        # print(self.images_index.shape)
        # print(self.given_label_matrix_index.shape)

    def init_index(self):
        self.return_truelabel=False
        self.true_labels_index=self.true_labels
        self.images_index = self.images
        self.given_label_matrix_index = self.given_label_matrix

    def __len__(self):
        return len(self.images_index)
        
    def __getitem__(self, index):
        each_iamge=self.images_index[index]
        each_label = self.given_label_matrix_index[index]
        each_true_label = self.true_labels_index[index]
        # each_iamge = Image.fromarray(each_iamge)
        each_iamge = Image.fromarray(np.transpose(each_iamge, (1, 2, 0)))
        if self.return_truelabel:
            each_selcted_label=self.selected_label[index]
            return self.transform(each_iamge), each_label,each_true_label,each_selcted_label,index
        else:
            return self.transform(each_iamge), each_label,index