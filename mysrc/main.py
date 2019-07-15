from Dogdataset import DDS
from torch.utils.data import DataLoader
from network import DogModel
import torchvision
import torch.nn as nn
import torch.nn.functional
import torch
import pandas as pd
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
from PIL import Image
from thop import profile
import torchvision.models as models
from visdom import Visdom

from utils.cal_time import *

train_data = '../datasets/train'
testing_data = '../datasets/val'
label_data = '../datasets/labels.csv'
batch_size = 6
train_batch_size = 1
epoch = 20
lr = 0.001
use_gpu = True
df = pd.read_csv(label_data)
breed_arr = sorted(list(set(df['breed'].values)))
num_classes = 120


def train(tl, vl):
    viz = Visdom(env='test')
    x, y = 0, 0
    win = viz.line(X=np.array([x]), Y=np.array([y]))
    criterion = nn.CrossEntropyLoss()
    model = models.resnet152(pretrained=True)  # ---> input size must be 3x299x299
    print(model.fc.in_features)
    model.fc = nn.Linear(2048, num_classes)

    # model = DogModel()
    if torch.cuda.is_available() and use_gpu:
        print('GPU ready')
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #use smaller lr for pre-trained models
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
    accuracy_test = []
    for i in range(epoch):
        model.train()
        for iter, sample in enumerate(tl):  # iterates len(database)/batch_size
            # tic()
            if iter == len(tl) - 1:
                continue
            input = sample['Image'].cuda() if use_gpu else sample['Image']  # size of batch_size
            label = sample['IntLabel'].cuda() if use_gpu else sample['IntLabel']
            optimizer.zero_grad()  # clears
            output = model(input)  # output is a batch_size x n_class 2d arr
            score = torch.max(output, 1)[1]  # array of the indexes of the max output number in each row
            correct = int(torch.sum(score == label))
            loss = criterion(output, label)
            viz.line(X=np.array([iter]), Y=np.array([float(loss)]), win=win, update='append')
            print('epoch {} = iteration {} lr {} == > loss {:.3f} total {} accuracy {:.3f}% correct {}'.format(
                i, iter, optimizer.state_dict()['param_groups'][0]['lr'], loss.data, batch_size,
                100 * correct / len(input), correct))
            loss.backward()
            optimizer.step()
        # scheduler.step()
        # toc()
        eval_acc = val(vl, model)["Accuracy"]
        total_correct = val(vl, model)["Correct"]
        num_images = val(vl, model)["Total"]
        print('=====> eval accuracy :{}'.format(eval_acc))
        accuracy_test.append(eval_acc)
        if i == 0 or accuracy_test[-1] >= max(accuracy_test[:-1]):
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, '../ResNet/' + str(accuracy_test[-1])[:4] + '_' + str(total_correct) + '_' + str(num_images) + '.tar')


def val(vl, m):
    m.eval()  # use when trying to compare label and predictor
    count = 0
    for iter, sample in enumerate(vl):
        input = sample['Image'].cuda()
        # img_name = sample['Name']
        # str_label = sample['Breed']
        label = sample['IntLabel'].cuda()
        output = m(input)
        score = torch.max(output, 1)[1]
        # for id, i in enumerate(score):
        #     img = Image.open(img_name[id])
        #     plt.imshow(img)
        #     plt.text(0,0,'true is {} and predict is {}'.format(str_label[id], breed_arr[i]))
        #     plt.show()
        correct = int(torch.sum(score == label))
        count += correct
    info = {"Accuracy": count / (len(vl) * batch_size) * 100, "Correct": count, "Total": (len(vl) * batch_size)}
    return info


def inference():
    # model = torchvision.models.googlenet(pretrained=True)
    model = DogModel()
    checkpoint = torch.load('/home/ai/Desktop/project2/AlexCheckpoint/99.70703125.tar')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.cuda()

    test_root = '/home/ai/Desktop/project2/datasets/val'
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    for im in os.listdir(test_root):
        label = breed_arr.index(df[df['id'] == im.split('.')[0]]['breed'].values[0])
        im_name = os.path.join(test_root, im)
        img = Image.open(im_name)
        input = torch.unsqueeze(test_transform(img), dim=0).cuda()
        output = model(input)
        predict = torch.max(output, 1)[1]
        if breed_arr[label] != breed_arr[int(predict)]:
            _, indices = torch.sort(output, descending=True)
            percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
            [print('\n' + breed_arr[idx], percentage[idx].item()) for idx in indices[0][:5]]
            vis(img, breed_arr[label], breed_arr[int(predict)])

    # test_set = DDS(test_root, label_data, test_transform)
    # testDataLoader = DataLoader(test_set, batch_size=1)
    # for sample in testDataLoader:
    #     input, target = sample['Image'].cuda(), sample['IntLabel'].cuda()
    #     output = model(input)
    #     predict_int_label = torch.max(output, 1)[1]
    #     print(list(target), list(predict_int_label))


def main():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.TenCrop((299,299)),
        torchvision.transforms.ToTensor()
    ])
    train_set = DDS(train_data, label_data, train_transform)
    testing_set = DDS(testing_data, label_data, test_transform)
    train_dataLoader = DataLoader(train_set, batch_size, shuffle=True)
    testing_dataLoader = DataLoader(testing_set, batch_size)
    train(train_dataLoader, testing_dataLoader)
    # inference()


main()
