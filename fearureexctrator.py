import os
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

HAS_SK = True
import numpy as np


DATA_ROOT = 'D:/data/dog-breed-identification'
all_labels_df = pd.read_csv(os.path.join(DATA_ROOT,'labels.csv'))


breeds = all_labels_df.breed.unique()
breed2idx = dict((breed,idx) for idx,breed in enumerate(breeds))
idx2breed = dict((idx,breed) for idx,breed in enumerate(breeds))

# 添加到列表中
all_labels_df['label_idx'] = [breed2idx[b] for b in all_labels_df.breed]


class DogDataset(Dataset):
    def __init__(self, labels_df, img_path, transform=None):
        self.labels_df = labels_df
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return self.labels_df.shape[0]

    def __getitem__(self, idx):
        image_name = os.path.join(self.img_path, self.labels_df.id[idx]) + '.jpg'
        img = Image.open(image_name)
        label = self.labels_df.label_idx[idx]

        if self.transform:
            img = self.transform(img)
        return img, label

IMG_SIZE = 224         # resnet50的输入是224的所以需要将图片统一大小
BATCH_SIZE= 128        # 这个批次大小需要占用(4.6-5g)/2的显存，如果内存超过10G可以改为512
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])


dataset_names = ['train', 'valid']
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_split_idx, val_split_idx = next(iter(stratified_split.split(all_labels_df.id, all_labels_df.breed)))
train_df = all_labels_df.iloc[train_split_idx].reset_index()
val_df = all_labels_df.iloc[val_split_idx].reset_index()
print(len(train_df))
print(len(val_df))


image_transforms = {'train':train_transforms, 'valid':val_transforms}

train_dataset = DogDataset(train_df, os.path.join(DATA_ROOT,'train'), transform=image_transforms['train'])
val_dataset = DogDataset(val_df, os.path.join(DATA_ROOT,'train'), transform=image_transforms['valid'])
image_dataset = {'train':train_dataset, 'valid':val_dataset}

image_dataloader = {x:DataLoader(image_dataset[x],batch_size=BATCH_SIZE,shuffle=True,num_workers=0) for x in dataset_names}
dataset_sizes = {x:len(image_dataset[x]) for x in dataset_names}


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


model_ft = models.resnet50(pretrained=True)      # 这里自动下载官方的预训练模型，并且

for param in model_ft.parameters():
    param.requires_grad = False                  # 将所有的参数层进行冻结

print(model_ft.fc)                               # 这里打印下全连接层的信息
num_fc_ftr = model_ft.fc.in_features             # 获取到fc层的输入
model_ft.fc = nn.Linear(num_fc_ftr, len(breeds)) # 定义一个新的FC层
model_ft=model_ft.to(DEVICE)                     # 放到设备中

print(model_ft.conv1.in_channels)
print(model_ft.conv1.out_channels)
print(model_ft.layer2[0].conv1.in_channels)
print("========================================")
print(model_ft)                                  # 最后再打印一下新的模型


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{'params':model_ft.fc.parameters()}],
                             lr=0.001)           # 指定 新加的fc层的学习率

def train(model,device, train_loader, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x,y= data
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_hat= model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        print(batch_idx)
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            x,y= data
            x=x.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += criterion(y_hat, y).item() # sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1]    # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()

            # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            # plot_only = 500
            # low_dim_embs = tsne.fit_transform(fet[:plot_only, :])
            # labels = y.cpu().numpy()[:plot_only]
            # plot_with_labels(low_dim_embs, labels)


    test_loss /= len(test_loader.dataset)
    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_dataset),
        100. * correct / len(val_dataset)))

# for epoch in range(1, 10):
#     train(model=model_ft,device=DEVICE, train_loader=image_dataloader["train"],epoch=epoch)
#     test(model=model_ft, device=DEVICE, test_loader=image_dataloader["valid"])


# print(model_ft.avgpool.input.size())


# model_ft=model_ft.to(DEVICE)
# in_list= [] # 这里存放所有的输出
# labels_list = []
# def hook(module, input, output):
#     #input是一个tuple代表顺序代表每一个输入项，我们这里只有一项，所以直接获取
#     #需要全部的参数信息可以使用这个打印
#     for val in input:
#        print("input val:",val)
#     for i in range(input[0].size(0)):
#         in_list.append(input[0][i].cpu().numpy())
#
# # 在相应的层注册hook函数，保证函数能够正常工作，我们这里直接hook 全连接层前面的pool层，获取pool层的输入数据，这样会获得更多的特征
# model_ft.avgpool.register_forward_hook(hook)
#
# # 开始获取输出，这里我们因为不需要反向传播，所以直接可以使用no_grad嵌套
# with torch.no_grad():
#     for batch_idx, data in enumerate(image_dataloader["train"]):
#         x,y= data
#         labels_list += y.numpy().tolist()
#         x=x.to(DEVICE)
#         y=y.to(DEVICE)
#         y_hat = model_ft(x)
#
#
##保存到diss
# features=np.array(in_list)
# np.save("features",features)
# labels_s=np.array(labels_list)
# np.save("labels",labels_s)
#

##加载保存文件并进行TSNE
fet = np.load('./features.npy')
labels1 = np.load('./labels.npy')

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
t = features.shape[0]
ft_tensor = torch.tensor(features)
fets = ft_tensor.view(ft_tensor.size()[0],-1) #tensor采有view()函数
low_dim_embs = tsne.fit_transform(fets[:plot_only, :])
# labels11 = labels1.cpu().numpy()[:plot_only] #labels1已经是numpy了
labels11 = labels1[:plot_only]
plot_with_labels(low_dim_embs, labels11)


# j =0
