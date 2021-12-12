import torch
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
from PIL import Image
import json


class CIFARDataset(torchvision.datasets.CIFAR10):
    def __init__(self, **kwargs):
        super(CIFARDataset, self).__init__(**kwargs)
        self.transformed_data = None
        if self.transform is not None:
            self.transformed_data = [self.transform(Image.fromarray(img)) for img in self.data]

    def __getitem__(self, index: int):
        img = self.transformed_data[index] if self.transformed_data is not None else self.data[index]
        target = self.targets[index]
        return img, target


def cifar_dataloaders(root="./cifar10", index_path="./index.json", batch_size=128, show=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    ################################################################################################
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_data_global = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_train)
    test_data_global = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    ################################################################################################
    file_ = open(index_path, 'r')
    context = json.load(file_)
    file_.close()


    trainloaders = {}
    for i in range(len(context.keys())):
        trainloaders[i] = torch.utils.data.DataLoader(trainset,
                                                      batch_size=batch_size,
                                                      sampler=SubsetRandomSampler(context[str(i)]))
    
    testloaders = {}
    for i in range(len(context.keys())):
        testloaders[i] = test_data_global

        
    class_num = 10
    train_data_num = len(trainset)
    test_data_num = len(testset)
    
    train_data_local_num_dict = {}
    for k in trainloaders.keys():
        train_data_local_num_dict[k] = len(trainloaders[k].dataset)
        

    if show:
        for j in range(len(context.keys())):
            ans = [0 for i in range(class_num)]
            for i in context[str(j)]:
                ans[trainset.targets[i]] += 1
            print("client: {} , {}, sum: {}".format(j, ans, sum(ans)))


    dataset = [train_data_num, 
               test_data_num, 
               train_data_global,
               test_data_global,
               train_data_local_num_dict,
               trainloaders,
               testloaders,
               class_num
              ]
    return dataset
# [
#     train_data_num, 
#     test_data_num, 
#     train_data_global, 
#     test_data_global,
#     train_data_local_num_dict, 
#     train_data_local_dict, 
#     test_data_local_dict, 
#     class_num
# ]