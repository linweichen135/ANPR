import cv2
import numpy as np
import datetime
from PIL import Image


##################    
#  general usage
##################

class Coord :
    def __init__ (self, X, Y):
        self.X = X
        self.Y = Y

class Shape :
    def __init__ (self, width, height):
        self.width = width
        self.height = height

def addSrcImage(img, src, coord, resize=False, shape=None, mode='RGB') :    # PNG format is required for RGBA image
    if(coord.X < 0 or coord.Y < 0):
        raise Exception('Error: image underflow')
    img_append = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if(resize == True):
        img_append = cv2.resize(img_append, (shape.width, shape.height), interpolation=cv2.INTER_NEAREST)
    destX = coord.X + img_append.shape[1]
    destY = coord.Y + img_append.shape[0]
    if(destX > img.shape[1] or destY > img.shape[0]):
        raise Exception('Error: image overflow')
    if(mode == 'RGB'):
        img[coord.Y:destY, coord.X:destX, :] = img_append
    elif(mode == 'RGBA'):
        for X in range(img_append.shape[1]):
            for Y in range(img_append.shape[0]):
                if(img_append[Y,X,3] == 255):
                    img[coord.Y+Y, coord.X+X, :] = img_append[Y, X, 0:3]
    return img
    
def addImage(img, img_append, coord, resize=False, shape=None, mode='RGB') :
    if(coord.X < 0 or coord.Y < 0):
        raise Exception('Error: image underflow')
    if(resize == True):
        img_append = cv2.resize(img_append, (shape.width, shape.height), interpolation=cv2.INTER_NEAREST)
    destX = coord.X + img_append.shape[1]
    destY = coord.Y + img_append.shape[0]
    if(destX > img.shape[1] or destY > img.shape[0]):
        raise Exception('Error: image overflow')
    if(mode == 'RGB'):
        img[coord.Y:destY, coord.X:destX, :] = img_append
    elif(mode == 'RGBA'):
        for X in range(img_append.shape[1]):
            for Y in range(img_append.shape[0]):
                if(img_append[Y,X,3] == 255):
                    img[coord.Y+Y, coord.X+X, :] = img_append[Y, X, 0:3]
    return img
    
class Displayer :    # unsupported for RGBA image
    def __init__ (self):
        self.windowsInfo = []
    def setWindow(self, img, coord, shape):
        self.windowsInfo.append( [coord,shape] )
        img_append = np.empty( (shape.height, shape.width, 3) )
        img_append.fill(255)
        addImage(img, img_append, coord)
    def display(self, img, img_append, windowIdx):
        addImage(img, img_append, self.windowsInfo[windowIdx][0], True, self.windowsInfo[windowIdx][1])
    def paint(self, img, windowIdx, color):
        X = self.windowsInfo[windowIdx][0].X
        Y = self.windowsInfo[windowIdx][0].Y
        width = self.windowsInfo[windowIdx][1].width
        height= self.windowsInfo[windowIdx][1].height
        img[Y:Y+height, X:X+width,:] = color
        
def PILtoMat(img_pil) :
    img = np.array(img_pil)
    if len(img.shape) != 3 :
        img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR )
    img = img[:, :, ::-1]
    return img


####################    
#  specific usage
####################

def setup_title(img) :    # img is expected to be 1920 * 1080
    cv2.putText(img, 'ADAR Lab', (1500,70), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255,0,255), 8)
    img = addSrcImage(img, 'pictures/decorate.png', Coord(1350,5), True, Shape(130,120), 'RGBA')
    cv2.putText(img, 'Copyright 2019', (1542,115), cv2.FONT_HERSHEY_DUPLEX, 1.15, (255,0,255), 4)
    return img

def writePlate(img, text) :
    cv2.putText(img, text, (1250,915), cv2.FONT_HERSHEY_DUPLEX, 2.5, (237,149,100), 8) # 744 872
    cv2.putText(img, str(datetime.datetime.now()), (1161,965), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (237,149,100), 4)
    # cv2.putText(img, 'entry', (1406,1007), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (237,149,100), 4)
    return img