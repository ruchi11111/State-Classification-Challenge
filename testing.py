import os, random, math,glob
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#from workspace_utils import active_session
from PIL import Image
from collections import OrderedDict
import json
from args import get_parser

data_dir = 'data'
dataset_split_path= os.path.join(data_dir, "test")
image_path = glob.glob(os.path.join(dataset_split_path, '*.jpg'))
image_path.sort()

idx_to_class = {0:'creamy_paste',1:'diced',2:'floured',3:'grated',4:'juiced',5:'julienne',6:'mixed',7:'other',8:'peeled',9:'sliced',10:'whole'}

data={}
temp_data ={}
model = torch.load('model.pth')


def process_image(image):
  #for i in image:
    pil_image = Image.open(image)
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(180),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
    ])
    
    img = image_transforms(pil_image)
    #emptylist.append(img)
    return img

#process_image(image_path) 

#print(emptylist)

def predict(image, model, topk=5):
    model.eval()
    model.cpu()
    img = process_image(image)
    #for img in path_list:
    img = img.unsqueeze_(0)
    img = img.float()
    with torch.no_grad():
        output = model.forward(img)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()
    for each in classes.cpu().numpy()[0]:
        top_classes = idx_to_class[each]  
    #print('Top Classes: ', top_classes)
    #print('Top Probs:', top_prob)
    return top_classes

for item in image_path:
  tpclass = predict(item,model,5)
  temp_data= {item:tpclass}
  #print(temp_data)
  data.update(temp_data)
#print(data)

with open('test_outputs.json','w') as out:
  json.dump(data,out)

print("Done with testing!!!!!")








