import torch
from torch import nn
from model import Net
from torchvision import transforms
from argparse import ArgumentParser
from PIL import Image
import timeit
import os


# settings
#--------------------------------------------------------------------------------------------------
MODEL_WEIGHTS_PATH = 'ocr_modelweights.pth'
#--------------------------------------------------------------------------------------------------


# global variables declaration
#--------------------------------------------------------------------------------------------------
device = torch.device(('cuda:0') if torch.cuda.is_available() else "cpu")

label_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
               'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
               'Y', 'Z']
#--------------------------------------------------------------------------------------------------


# class ocr definition
#--------------------------------------------------------------------------------------------------
class ocr() :
    def __init__(self) :
        self.model = Net()
        self.model.load_state_dict( torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')) )
        self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    def recognize(self, img_pil) :
        img_pil = img_pil.convert('L')
        image = self.transform(img_pil)
        image = image.to(device)
        image = image.unsqueeze(0)
        output = self.model(image)
        _, predict = torch.max(output, 1)
        return label_chars[predict]
#--------------------------------------------------------------------------------------------------


# testing code
#--------------------------------------------------------------------------------------------------
if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('testdata_directory', type=str, help='the path of testdata directory,\
                         testdata must be named beginning with label.')
    args = parser.parse_args()
    OCR = ocr()
    tstart = timeit.default_timer()
    filenum = 0
    accuracy = 0
    for root, dirs, files in os.walk(args.testdata_directory) :  
        for filename in files:
            img_pil = Image.open(args.testdata_directory + '/' + filename)
            predict = OCR.recognize(img_pil)
            print(filename + ': ' + predict) 
            filenum += 1
            if predict == filename[0] :
                accuracy += 1
    tend = timeit.default_timer()
    print('Accuracy: ' + str(float(accuracy)/filenum*100) + '%')
    print('Average process time per image: ' + str((tend - tstart) / filenum))
    
    # models information
    from torchsummary import summary
    from thop import profile
    summary(OCR.model, (1,32,32))
    flops, params = profile(OCR.model, inputs=(torch.randn(1,1,32,32).to(device),))
    print("FLOPS: ", flops, " / Params: ", params)
# --------------------------------------------------------------------------------------------------