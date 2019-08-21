from __future__ import print_function
import csv
import torch
import datetime
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import time


# In[2]:


def default_loader(path):
    im = Image.open(path).convert('L')
    im= im.resize((34,34),Image.BILINEAR)
    
    return im


# In[3]:


def f(x):
    if x =='0':
        return 0
    if x =='1':
        return 1
    if x =='2':
        return 2
    if x =='3':
        return 3
    if x =='4':
        return 4
    if x =='5':
        return 5
    if x =='6':
        return 6
    if x =='7':
        return 7
    if x =='8':
        return 8
    if x =='9':
        return 9
    if x =='A':
        return 10
    if x =='B':
        return 11
    if x =='C':
        return 12
    if x =='D':
        return 13
    if x =='E':
        return 14
    if x =='F':
        return 15
    if x =='G':
        return 16
    if x =='H':
        return 17
    if x =='J':
        return 18
    if x =='K':
        return 19
    if x =='L':
        return 20
    if x =='M':
        return 21
    if x =='N':
        return 22
    if x =='P':
        return 23
    if x =='Q':
        return 24
    if x =='R':
        return 25
    if x =='S':
        return 26
    if x =='T':
        return 27
    if x =='U':
        return 28
    if x =='V':
        return 29
    if x =='W':
        return 30
    if x =='X':
        return 31
    if x =='Y':
        return 32
    if x =='Z':
        return 33


# In[4]:


class myImageFloder(data.Dataset):
    
    def __init__(self,root,image_path,label_path,transform = None,target_transform=None,loader = default_loader):
        File_image = open(image_path,'r')
        File_label = open(label_path,'r')
        
        imgs = []
        img_names = []
        label_names = []
        
        for line in File_image.readlines():
            cls = line.strip()
            img_name = cls
            img_names.append(img_name)
            if os.path.isfile(root+img_name):
                if Image.open(root+img_name):
                    
                    imgs.append(img_name)
                                
        for line in File_label.readlines():
            cls = line.strip()
            label_name = cls
            label_names.append(f(label_name))
        
        self.root = root
        self.imgs = imgs
        self.img_names = img_names
        self.lable_names = label_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self,index):
        
        img_name= self.imgs[index]
        label = self.lable_names[index]
        
        
        #label = np.zeros(34)
        #label[int(f(label_name))]=1
        
        img = self.loader(self.root+img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img,label,img_name
    
    def __len__(self):
        
        return len(self.imgs)


# In[5]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3)
        self.conv3_drop = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(200, 34)
        
    def forward(self, x):  # 34x34x1
        # CONV1
        x = self.conv1(x)  # 32x32x2
        x = F.max_pool2d(x,2)  # 16x16x2
        x = F.relu(x)
        # CONV2
        x = self.conv2(x)  # 14x14x4
        x = F.max_pool2d(x,2)  # 7x7x4
        x = F.relu(x)
        x = self.conv2_drop(x)
        # CONV3
        x = self.conv3(x)  # 5x5x8
        x = F.relu(x)
        x = self.conv3_drop(x)
        # FLATTEN
        x = x.view(x.size(0),-1)  # 200x1
        # FC1
        x = self.fc1(x)  # 34x1
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


model = Net().to(device)


# In[8]:


learning_rate = 0.001
epochs = 15
batch_size = 128


# In[9]:


loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr =learning_rate)


# In[10]:


train_data= myImageFloder(root = '/mnt/ssdraid/ocr_traindata/',
                                image_path = '/mnt/ssdraid/ocr_traindata/image_train.txt' ,
                                label_path = '/mnt/ssdraid/ocr_traindata/label_train.txt' ,
                                transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          ] ))
train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size = batch_size,shuffle = True, num_workers = 4)
test_data_new = myImageFloder(root = '/mnt/ssdraid/ocr_traindata/',
                                image_path = '/mnt/ssdraid/ocr_traindata/image_train.txt' ,
                                label_path = '/mnt/ssdraid/ocr_traindata/label_train.txt' ,
                                transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          ]))
test_loader = torch.utils.data.DataLoader(dataset = test_data_new,batch_size = 128, num_workers = 4)


# In[13]:


total_step = len(train_loader)
print(total_step)
StartTime_train = time.time()
model.train()
acc = []

for epoch in range (epochs):
    for indx , (images,labels,img_name) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        #labels = labels.type(torch.cuda.LongTensor)
        #Forward pass
        outputs = model(images)
        
        loss = loss_f(outputs,labels)
        
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(epoch ==25):
          for param_group in optimizer.param_groups:
              param_group['lr'] = learning_rate/10
        if(indx+1)%100==0:
            print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f}'.format(epoch+1, epochs, indx+1, total_step, loss.item()))
    model.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      error = []
      for images,labels,names in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = loss_f(outputs,labels)
        _,predicted = torch.max(outputs.data,1)
        total +=labels.size(0)
        correct+=(predicted == labels).sum().item()  
      acc.append(100*correct/total)
      print('Test_new Accuracy of the model on the test imgs:{} %,Loss:{:.4f}'.format(acc[epoch],loss.item()))
      if(epoch == 0):
        flag = acc[epoch]
      if(epoch >0):
        if(acc[epoch]>flag):
          flag = acc[epoch]
          print('save higher accuracy model in epoch  ',epoch+1)
          torch.save(model, 'model')
    
EndTime_train = time.time()        
print('Train Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime_train-StartTime_train)))))







alp=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
error_matrix=np.zeros((35,35),dtype = int)
for i in range(34):
    error_matrix[i+1][0]=i
for i in range(34):
    error_matrix[0][i+1]=i
tStart = time.time()
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    error = []
    error.append(('image_name','predicted','label'))
    for images,labels,names in test_loader:
        images = images.to(device)
        labels = labels.to(device)
		
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total +=labels.size(0)
        correct+=(predicted == labels).sum().item()
        
        for i in range(predicted.size(0)):
              if(predicted[i]!=labels[i]):
                error_matrix[predicted[i]+1][labels[i]+1]+=1
                predicted_num = predicted[i].cpu().numpy()
                labels_num = labels[i].cpu().numpy()
                error.append((names[i],alp[predicted_num],alp[labels_num]))
        
    print('Accuracy :{} %'.format(100*correct/total))
tEnd = time.time()
#print ('Time: {:f}'.format(tEnd - tStart))
print('Test Time Usage: ', str(datetime.timedelta(seconds=int(round(tEnd-tStart)))))
print ('fps: {:f}'.format(len(test_data_new)/(tEnd - tStart)))

new_list = error_matrix.tolist()

for i in range(34):
    new_list[i+1][0]=alp[i]
for i in range(34):
    new_list[0][i+1]=alp[i]

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(new_list)
    writer.writerows(error)


np.save('conv1_w.npy', model.conv1.weight.cpu().detach().numpy())
np.save('conv2_w.npy', model.conv2.weight.cpu().detach().numpy())
np.save('conv3_w.npy', model.conv3.weight.cpu().detach().numpy())
np.save('fc1_w.npy', model.fc1.weight.cpu().detach().numpy())

np.save('conv1_b.npy', model.conv1.bias.cpu().detach().numpy())
np.save('conv2_b.npy', model.conv2.bias.cpu().detach().numpy())
np.save('conv3_b.npy', model.conv3.bias.cpu().detach().numpy())
np.save('fc1_b.npy', model.fc1.bias.cpu().detach().numpy())
