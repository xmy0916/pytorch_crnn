import torch.utils.data as data # 加载torch的数据加载器
import numpy as np
import time
import cv2
import sys
import os
sys.path.append(os.getcwd())
import argparse
import yaml
import model.model as crnn
from easydict import EasyDict as edict
import torch
import torch.optim as optim
# 实现模板类
class OCRDataset(data.Dataset):
    def __init__(self,config,is_train=True):
        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W
        self.dataset_name = config.DATASET.DATASET
        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)
        char_file = config.DATASET.CHAR_FILE
        with open(char_file, 'rb') as file:
           lines = file.readlines()
           self.char_dict = {num: char.strip().decode('utf-8', 'ignore') for num, char in enumerate(lines)}
           self.char_dict2 = {char.strip().decode('utf-8', 'ignore'):num for num, char in enumerate(lines)}
        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']
        txt_file = os.path.join(self.root,txt_file)
        # convert name:indices to name:string
        self.labels = []
        print(self.char_dict2)
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imgname = c.split('\t')[0]
                string = c.split('\t')[1].replace("\n","")
                self.labels.append({imgname: string})

        print("load {} images!".format(self.__len__())) 

    def __len__(self):
        # 实现模板方法
        return len(self.labels)

    def __getitem__(self,idx):
        img_name = list(self.labels[idx].keys())[0]
        img = cv2.imread(os.path.join(self.root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape
        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))
        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx

def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label

def get_optimizer(config, model):

    optimizer = None

    if config.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
        )
    elif config.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            # alpha=config.TRAIN.RMSPROP_ALPHA,
            # centered=config.TRAIN.RMSPROP_CENTERED
        )

    return optimizer
def encode(char_dict,text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                index = char_dict[char]
                result.append(index)
        text = result
        #print(result)
        return (torch.IntTensor(text), torch.IntTensor(length))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

if __name__ == '__main__':
    config = parse_arg()    
    train_dataset = OCRDataset(config)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    config.MODEL.NUM_CLASSES = len(train_dataset.char_dict2)
    model = crnn.get_crnn(config)
    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)
    last_epoch = config.TRAIN.BEGIN_EPOCH
    criterion = torch.nn.CTCLoss()
    optimizer = get_optimizer(config, model)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    model_info(model)
    for epoch in range(last_epoch,config.TRAIN.END_EPOCH):
      batch_time = AverageMeter()
      data_time = AverageMeter()
      losses = AverageMeter()
      end = time.time()
      model.train()
      for i, (inp, idx) in enumerate(train_loader):
          data_time.update(time.time() - end)
          labels = get_batch_label(train_dataset, idx) # 一个batch的标签  
          inp = inp.to(device)
          # print("inp",inp[0].cpu().detach().numpy(),inp[0].cpu().detach().numpy().shape)
          # 前馈
          # print("forward")
          preds = model(inp).cpu()
          # print("pred",preds.detach().numpy(),preds.detach().numpy().shape)
          # exit(-1)
          #计算loss
          batch_size = inp.size(0)
          text, length = encode(train_dataset.char_dict2,labels)
          preds_size = torch.IntTensor([preds.size(0)] * batch_size)
          loss = criterion(preds, text, preds_size, length)
          # 反馈
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          losses.update(loss.item(), inp.size(0))
          batch_time.update(time.time()-end)
          if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

          end = time.time()
