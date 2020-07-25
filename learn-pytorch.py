import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim #最適化関数
import torch.nn.functional as F
import torch.utils as utils
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data

from torch.autograd import Variable #自動微分機能のimport

from PIL import Image
import math
import matplotlib.pyplot as plt

class SketchDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform = None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.label[idx]
        return self.transform(out_data), out_label

class SketchModel(nn.Module):
    """
    https://github.com/HPrinz/sketch-recognition
    input size: (225. 225)
    """
    def __init__(self):
        super(SketchModel, self).__init__()
        self.filter_num = 32
        self.output_num = 72
        self.reLU = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(1,self.filter_num*1,3)
        self.conv2 = nn.Conv2d(self.filter_num*1,self.filter_num*1,3)
        self.conv3 = nn.Conv2d(self.filter_num*1,self.filter_num*2,3)
        self.conv4 = nn.Conv2d(self.filter_num*2,self.filter_num*2,3)
        self.pool = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(self.filter_num*2*5*5, self.filter_num * 8)
        self.linear2 = nn.Linear(self.filter_num*8, self.output_num)

    def forward(self, input):
        x = input.reshape(-1,1,32,32)
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x)

        x = x.view(-1, self.filter_num*2*5*5)
        print(x.shape)
        x = F.relu(self.linear1(x))
        print(x.shape)
        x = F.dropout(x)

        x = self.linear2(x)
        print(x.shape)
        x = F.log_softmax(x, dim=1)

        return x

    def flatten(self, x):
        bs = x.size()[0]
        return x.view(bs, -1)

    def imshow(self, img, name):
        # 50 個のデータを出力する
        print(img.shape)
        pool = image.shape[0]
        width = image.shape[2]
        height = image.shape[3]
        images = img.numpy().reshape(pool,width,height)
        print(images.shape)

        plt.figure()
        for i in range(pool):
            plt.subplot(4, 4, i+1)
            plt.subplots_adjust(hspace=0.1, wspace=0.1)
            plt.axis("off")
            plt.imshow(image[i].reshape(width, height))
        plt.savefig("image/" + name)
        plt.close()

    def imshow_conv(self, img, name):
        print(img.shape)
        pool = img.shape[1]
        width = img.shape[2]
        height = img.shape[3]
        graph_size = math.sqrt(pool)
        graph_size = graph_size+1 if graph_size > int(graph_size) else graph_size

        images = img.detach().numpy()[0]

        plt.figure()
        for i in range(pool):
            plt.subplots_adjust(hspace=0.1, wspace=0.1)
            plt.subplot(graph_size, graph_size, i+1)
            plt.axis("off")
            plt.imshow(images[i])
        plt.savefig("image/" + name)
        plt.close()


all_data = np.load("hiragana.npz")['arr_0'].reshape([-1, 32, 32]).astype(np.float32)
all_label = np.repeat(np.arange(72), 160)

trans = transforms.Compose([transforms.ToTensor()])
data_train, data_test, label_train, label_test = train_test_split(all_data, all_label, test_size=0.2)

data = SketchDataset(np.array(data_train), label_train, trans)
data_test = SketchDataset(np.array(data_test), label_test, trans)
data_train_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
data_test_loader  = torch.utils.data.DataLoader(data_test, batch_size=16, shuffle=True)

model = SketchModel()
#Loss関数の指定
criterion = nn.CrossEntropyLoss()

#Optimizerの指定
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loss = np.array([])
test_loss = np.array([])

for epoch in range(50):

    loss_per_epoch = 0.0
    acc = 0.0

    print(len(data_train_loader))
    for idx, (image, label) in enumerate(data_train_loader):
        model.train()
        optimizer.zero_grad()
        image, label = Variable(image), Variable(label)
        estimated = model(image)
        loss = criterion(estimated, label)
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss.data
        acc += torch.sum(label == torch.argmax(estimated, dim=1)).cpu().numpy()
    train_loss = np.append(train_loss, loss_per_epoch)
    print("epoch:     {%03d}, train_loss: {}".format(epoch, train_loss[-1] / len(data_train_loader)))
    print("train_acc: {}".format(acc/len(data_train)))

    loss_per_epoch = 0.0
    acc = 0.0

    for idx, (image, label) in enumerate(data_test_loader):
        model.eval()
        image, label = Variable(image), Variable(label)
        estimated = model(image)
        loss = criterion(estimated, label)
        loss_per_epoch += loss.data
        acc += torch.sum(label == torch.argmax(estimated, dim=1)).cpu().numpy()
    test_loss = np.append(train_loss, loss_per_epoch)
    print("epoch:     {%03d}, test_loss: {}".format(epoch, test_loss[-1] / len(data_test_loader)))
    print("train_acc: {}".format(acc/len(data_test)))
