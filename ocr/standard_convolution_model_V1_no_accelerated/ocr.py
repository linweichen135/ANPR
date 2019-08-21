import numpy as np
from PIL import Image
from argparse import ArgumentParser
import os
import timeit
import sys
np.set_printoptions(threshold=sys.maxsize)


class ocr() :
    
    def __init__(self) :
        self.img_size = 34
        self.stride = 2
        self.kernel_size = 3
        self.conv_layer_1 = 2
        self.conv_layer_2 = 4
        self.conv_layer_3 = 8
        self.fc_in = 200
        self.fc_out = 34
        self.folder = 'weights'
        self.conv1_w = np.load(self.folder + '/conv1_w.npy')
        self.conv2_w = np.load(self.folder + '/conv2_w.npy')
        self.conv3_w = np.load(self.folder + '/conv3_w.npy')
        self.fc1_w = np.load(self.folder + '/fc1_w.npy')
        self.conv1_b = np.load(self.folder + '/conv1_b.npy')
        self.conv2_b = np.load(self.folder + '/conv2_b.npy')
        self.conv3_b = np.load(self.folder + '/conv3_b.npy')
        self.fc1_b = np.load(self.folder + '/fc1_b.npy')
        self.alp = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

    def convolution(self, image, filt, bias, stride=1) :
        (n_f, n_c_f, f, _) = filt.shape    # filter dimensions / n_f: kernel numbers, n_c_f: kernel channel, f: kernel size
        n_c, in_dim, _ = image.shape       # image dimensions / n_c: image channel, in_dim: image size
        out_dim = int((in_dim - f) / stride) + 1    # calculate output dimensions    
        assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"     
        out = np.zeros( (n_f, out_dim, out_dim) )
        # convolve the filter over every part of the image, adding the bias at each step. 
        for curr_f in range(n_f) :
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                    out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                    curr_x += stride
                    out_x += 1
                curr_y += stride
                out_y += 1
        return out

    def maxpool(self, image, f=2, s=2) :
        # Downsample `image` using kernel size `f` and stride `s`
        n_c, h_prev, w_prev = image.shape
        h = int((h_prev - f)/s)+1
        w = int((w_prev - f)/s)+1
        downsampled = np.zeros((n_c, h, w))
        for i in range(n_c) :
            # slide maxpool window over each part of the image and assign the max value at each step to the output
            curr_y = out_y = 0
            while curr_y + f <= h_prev :
                curr_x = out_x = 0
                while curr_x + f <= w_prev: 
                    downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        return downsampled
        
    def recognize(self, img_pil) :
        img_pil = img_pil.convert('L')
        if img_pil.size != (self.img_size , self.img_size) :
            img_pil = img_pil.resize((self.img_size , self.img_size), Image.BILINEAR)
        img = np.asarray( img_pil, dtype=np.float32 ) / 255
        img = img.reshape(1, self.img_size, self.img_size)
        #### Conv1 ####
        conv1 = self.convolution(img, self.conv1_w, self.conv1_b)
        conv1_mp = self.maxpool(conv1)
        conv1_relu = np.maximum(conv1_mp, 0)
        #### Conv2 ####
        conv2 = self.convolution(conv1_relu, self.conv2_w, self.conv2_b)
        conv2_mp = self.maxpool(conv2)
        conv2_relu = np.maximum(conv2_mp, 0)
        #### Conv3 ####
        conv3 = self.convolution(conv2_relu, self.conv3_w, self.conv3_b)
        conv3_relu = np.maximum(conv3, 0)
        #### Flatten ####
        conv3_flat = conv3_relu.flatten()
        #### FC1 ####
        fc1 = np.zeros(self.fc_out)
        for out_c in range(self.fc_out) :
            for in_c in range(self.fc_in):
                fc1[out_c] += conv3_flat[in_c] * self.fc1_w[out_c][in_c]
            fc1[out_c] += self.fc1_b[out_c]
        return self.alp[np.argmax(fc1)]
        
    def recognizePlate(self, charsPIL_list) :    # designed for cython
        recogRslt = ''
        for idx in range( min(len(charsPIL_list), 7) ) :
            recogRslt += self.recognize( charsPIL_list[idx] )
        return recogRslt



if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('testdata_directory', type=str, help='the path of testdata directory, testdata must be named beginning with label.')
    args = parser.parse_args()
    OCR = ocr()
    print('weight')
    print(OCR.conv1_w.shape)
    print(OCR.conv2_w.shape)
    print(OCR.conv3_w.shape)
    print(OCR.fc1_w.shape)
    print('bias')
    print(OCR.conv1_b.shape)
    print(OCR.conv2_b.shape)
    print(OCR.conv3_b.shape)
    print(OCR.fc1_b.shape)
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
    print('accuracy: ' + str(accuracy/filenum*100) + '%')
    print('average process time per image: ' + str( (tend-tstart)/filenum ))