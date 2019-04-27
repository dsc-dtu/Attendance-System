import os
import cv2
import numpy as np
import io
from google.cloud import vision
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision.transforms import ToTensor

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img1_tuple = random.choice(self.imageFolderDataset.imgs)
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

#````````````````````````````````````````````````````````````````````````````````````````

net = SiameseNetwork()
net = torch.load('data/model/siamese_model.pth')

#`````````````````````````````````````````````````````````````````````````````````````````

def nearest_match(test_img_frame,i):
    for root, dirs, files in os.walk("data/faces/students"):
        l = dirs
        break
    e_d = 1000
    label = None

    folder_dataset_test = dset.ImageFolder(root="data/extra")
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            ,should_invert=False)

    test_dataloader = DataLoader(siamese_dataset,batch_size=1,shuffle=True)
    dataiter = iter(test_dataloader)
    x0,_,_ = next(dataiter)
    # x0 = Image.open('data/extra/'+str(i)+'.png')
    # x0 = x0.convert("L")

    for name in l:

        # folder_dataset_test = dset.ImageFolder(root="data/faces/students/"+name)
        # siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
        #                                         transform=transforms.Compose([transforms.Resize((100,100)),
        #                                                                       transforms.ToTensor()
        #                                                                     ])
        #                                        ,should_invert=False)

        # test_dataloader = DataLoader(siamese_dataset,batch_size=1,shuffle=True)
        # dataiter = iter(test_dataloader)

        x1 = Image.open('data/faces/students/'+name+'/'+'3.png')
        x1 = x1.convert("L")

        
        # x0 = transforms.Resize((100,100))(x0)
        x1 = transforms.Resize((100,100))(x1)

        # x0 = ToTensor()(x0).unsqueeze(0)
        x1 = ToTensor()(x1).unsqueeze(0)
    
        output1,output2 = net(Variable(x0),Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        if e_d>euclidean_distance:
            e_d = euclidean_distance
            label = name

    return label

#````````````````````````````````````````````````````````````````````````````````````````
cap = cv2.VideoCapture(0)
dataset_path = 'data/'

face_data =[]
labels=[]

class_id = 0
names = {}

while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    client = vision.ImageAnnotatorClient()
    cv2.imwrite("data/frame.jpg", frame) 
    path="data/frame.jpg"
    with io.open(path,'rb')  as image_file:
        content =image_file.read()
    image = vision.types.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    i = 0
    for face in faces:
        i+=1
        b =[]
        for vertex in face.bounding_poly.vertices:
            b.append(vertex)
        x_i=int(b[0].x)
        x_f=int(b[2].x)
        y_i=int(b[0].y)
        y_f=int(b[2].y)
        face_section = frame[y_i:y_f,x_i:x_f]
        face_section = cv2.resize(face_section,(100,100))
        cv2.imwrite(dataset_path+'extra/'+str(i)+'.png', face_section)
        pred_name = nearest_match(face_section,i)
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x_i,y_i),(x_f,y_f),(0,255,255),2)

    cv2.imshow("Faces",frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('x'):
        break
    
cap.release()
cv2.destroyAllWindows()


