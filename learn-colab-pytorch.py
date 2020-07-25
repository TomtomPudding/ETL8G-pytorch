from google.colab import drive

drive.mount("/content/drive")
%cd /content/drive/'My Drive'/MachineLearning/ETL8G
%ls -a

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
        self.linear1 = nn.Linear(self.filter_num*5*5*2, self.filter_num * 8)
        self.linear2 = nn.Linear(self.filter_num*8, self.output_num)

    def forward(self, input):
        x = input.reshape(-1,1,32,32)
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = F.dropout(self.pool(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.dropout(x)

        x = x.view(-1, self.filter_num*5*5*2)
        x = F.relu(self.linear1(x))
        x = F.dropout(x)

        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def flatten(self, x):
        bs = x.size()[0]
        return x.view(bs, -1)


all_data = np.load("hiragana.npz")['arr_0'].reshape([-1, 32, 32]).astype(np.float32)
all_label = np.repeat(np.arange(72), 160)

trans = transforms.Compose([transforms.ToTensor()])
data_train, data_test, label_train, label_test = train_test_split(all_data, all_label, test_size=0.2)
data = SketchDataset(np.array(data_train), label_train, trans)
data = SketchDataset(np.array(data_train), label_train, trans)
data_test = SketchDataset(np.array(data_test), label_test, trans)
data_train_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
data_test_loader  = torch.utils.data.DataLoader(data_test, batch_size=16, shuffle=True)

model = SketchModel().cuda()
criterion = nn.CrossEntropyLoss().cuda()
# optimizer = optim.Adadelta(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loss = np.array([])
train_acc = np.array([])
test_loss = np.array([])
test_acc = np.array([])

for epoch in range(400):

    loss_per_epoch = 0.0
    acc = 0.0
    print("Epoch {:03d}/400".format(epoch+1))

    for idx, (image, label) in enumerate(data_train_loader):
        model.train()
        optimizer.zero_grad()
        image, label = image.cuda().float(), label.cuda().long()
        estimated = model(image)
        loss = criterion(estimated, label)
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss.storage()[0]
        acc += torch.sum(label == torch.argmax(estimated, dim=1)).cpu().numpy()
    train_loss = np.append(train_loss, loss_per_epoch)
    train_acc = np.append(train_acc, acc/len(data_train))
    print("loss: {:5g}, acc: {:.5g}".format(train_loss[-1] / len(data_train_loader), acc/len(data_train)))

    loss_per_epoch = 0.0
    acc = 0.0

    for idx, (image, label) in enumerate(data_test_loader):
        model.eval()
        image, label = image.cuda().float(), label.cuda().long()
        estimated = model(image)
        loss = criterion(estimated, label)
        loss_per_epoch += loss.storage()[0]
        acc += torch.sum(label == torch.argmax(estimated, dim=1)).cpu().numpy()
    test_loss = np.append(test_loss, loss_per_epoch)
    test_acc = np.append(test_acc, acc/len(data_test))
    print("loss: {:5g}, acc: {:5g}".format(test_loss[-1] / len(data_test_loader), acc/len(data_test)))


# import matplotlib.pyplot as plt
# x = np.arange(0, 400, 1) + 1

# plt.plot(x, train_loss, label="train")
# plt.plot(x, test_loss, label="test")
# plt.ylim(0.0, 1.0)
# plt.title("loss")
# plt.savefig("loss.png")
# plt.close()

# plt.plot(x, train_acc, label="train")
# plt.plot(x, test_acc, label="test")
# plt.ylim(0.0, 1.0)
# plt.title("accuracy")
# plt.savefig("result.png")
# plt.close()

# from sklearn.externals import joblib
# joblib.dump(model, 'digits.pkl')
