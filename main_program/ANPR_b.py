import multiprocessing
from multiprocessing import Queue
from argparse import ArgumentParser
import FuncAndClass
from FuncAndClass import Coord, Shape, Displayer
from correct import correct
import cv2
import timeit
import datetime
from PIL import Image
import location_b
from ocr import ocr
import numpy as np


def REC_proc(que_c2s_req, que_c2s_rslt, que_s2c, key, without_ocr) :
    print('[REC] start process - recognization')
    OCR = ocr()
    while True :
        que_c2s_req.put('REQUEST')
        content = que_s2c.get()
        if content == 'EXIT' :
            break
        img_pil = content
        licplate_pil, imm1_pil, imm2_pil, imm4_pil, success, num_word, charsPIL_list, imm3_pil = location_b.location(img_pil, blur=True)
        que_c2s_rslt.put(success)
        if success :
            que_c2s_rslt.put(imm1_pil)
            que_c2s_rslt.put(imm2_pil)
            que_c2s_rslt.put(imm3_pil)
            que_c2s_rslt.put(imm4_pil)
            que_c2s_rslt.put(licplate_pil)
            que_c2s_rslt.put(charsPIL_list)
            que_c2s_rslt.put(num_word)
            if not without_ocr :
                pred_list = []
                for idx in range( min(len(charsPIL_list), 7) ) :
                    pred_list.append( OCR.recognize( charsPIL_list[idx] ) )
                recogRslt = correct(pred_list)
                que_c2s_rslt.put(recogRslt)
    print('[REC] terminate process')
    unlock(key)
    
    
    
def clean(queue) :
    while queue.qsize() != 0 :
        queue.get()
        
        
        
def lock(key) :
    key.get()
    
    
    
def unlock(key) :
    key.put(0)
        
        
    
if __name__ == '__main__' :

    # parser ---------------------------------------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--fps', help='enable the function of fps counting', action='store_true')
    parser.add_argument('--without_ocr', help='disable the function of OCR', action='store_true')
    parser.add_argument('--video', nargs=1, help='input from a video')
    parser.add_argument('camera_number', type=int, nargs='?', default=0, help='the number of camera which is going to be opened')
    args = parser.parse_args()
    #-----------------------------------------------------------------------------------------------------
    
    print('[MAIN] start process - control and display')
   
    # setup display interface ----------------------------------------------------------------------------
    cap = cv2.VideoCapture(args.camera_number)
    if args.video :
        print(args.video[0])
        cap = cv2.VideoCapture(args.video[0])
    if not cap.isOpened() :
        print('[MAIN] failed to open camera')
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #background = cv2.imread('pictures/background.jpg', cv2.IMREAD_COLOR)    # use background image
    background = np.zeros((1080,1920,3), dtype=np.uint8)
    background[:,:] = (255,255,255)
    img = background.copy()
    img = FuncAndClass.setup_title(img)
    disp = Displayer()
    cv2.putText(img, 'live image', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(15,40), Shape(640,400))
    disp.paint(img, 0, (0,0,0))
    cv2.putText(img, 'input image', (670,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(670,40), Shape(640,400))
    disp.paint(img, 1, (0,0,0))
    cv2.putText(img, 'imm1', (15,470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(15,480), Shape(480,270))
    disp.paint(img, 2, (0,0,0))
    cv2.putText(img, 'imm2', (510,470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(510,480), Shape(480,270))
    disp.paint(img, 3, (0,0,0))
    cv2.putText(img, 'imm3', (15,780), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(15,790), Shape(480,270))
    disp.paint(img, 4, (0,0,0))
    cv2.putText(img, 'imm4', (510,780), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(510,790), Shape(480,270))
    disp.paint(img, 5, (0,0,0))
    cv2.putText(img, 'license plate', (1005,470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(1005,480), Shape(305,100))
    disp.paint(img, 6, (0,0,0))
    cv2.putText(img, 'splitted characters', (1005,610), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(1005,620), Shape(80,160))
    disp.paint(img, 7, (0,0,0))
    disp.setWindow(img, Coord(1100,620), Shape(80,160))
    disp.paint(img, 8, (0,0,0))
    disp.setWindow(img, Coord(1195,620), Shape(80,160))
    disp.paint(img, 9, (0,0,0))
    disp.setWindow(img, Coord(1290,620), Shape(80,160))
    disp.paint(img, 10, (0,0,0))
    disp.setWindow(img, Coord(1385,620), Shape(80,160))
    disp.paint(img, 11, (0,0,0))
    disp.setWindow(img, Coord(1480,620), Shape(80,160))
    disp.paint(img, 12, (0,0,0))
    disp.setWindow(img, Coord(1575,620), Shape(80,160))
    disp.paint(img, 13, (0,0,0))
    disp.setWindow(img, Coord(1670,620), Shape(80,160))
    disp.paint(img, 14, (0,0,0))
    cv2.putText(img, 'recognition result', (1005,810), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    disp.setWindow(img, Coord(1005,820), Shape(900,230))
    disp.paint(img, 15, (0,0,0))
    cv2.rectangle(img, (1330,140), (1900,600), (255,0,0), 4)
    cv2.putText(img, 'current time:', (1345,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,85), 2)
    cv2.putText(img, str( datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") ), (1525,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,85), 2)
    #-----------------------------------------------------------------------------------------------------
    
    # start REC process ----------------------------------------------------------------------------------
    que_c2s_req = Queue()
    que_c2s_rslt = Queue()
    que_s2c = Queue()
    key = Queue()
    proc_rec = multiprocessing.Process(target=REC_proc, args=(que_c2s_req, que_c2s_rslt, que_s2c, key, args.without_ocr))
    proc_rec.start()
    # ----------------------------------------------------------------------------------------------------
   
    # main process  --------------------------------------------------------------------------------------
    tstartMain = timeit.default_timer()
    if(args.fps) :
        tstartFps = tstartMain
        frameCount = 0
    request = False
    success = False
    while True :
        ret, frame = cap.read()
        if not ret :
            break
        disp.display(img, frame, 0)
        if not que_c2s_req.empty() and not request :                      # request from REC detected
            request = True
            que_c2s_req.get()
            inputImg = frame
            inputImg_pil = cv2.cvtColor( inputImg, cv2.COLOR_BGR2RGB )    # convert "frame" to PIL format
            inputImg_pil = Image.fromarray(inputImg_pil)
            que_s2c.put(inputImg_pil)                                     # send the latest frame captured from the camera
            if(args.fps) :                   
                frameCount = frameCount + 1
        if not que_c2s_rslt.empty() and not success :                     # results from REC detected
            success = que_c2s_rslt.get()
            if not success :
                request = False
        qsize_rslt = 8
        if args.without_ocr :
            qsize_rslt = 7
        if success and que_c2s_rslt.qsize() == qsize_rslt :
            request = False
            success = False
            imm1_pil = que_c2s_rslt.get()
            imm2_pil = que_c2s_rslt.get()
            imm3_pil = que_c2s_rslt.get()
            imm4_pil = que_c2s_rslt.get()
            licplate_pil = que_c2s_rslt.get()
            charsPIL_list = que_c2s_rslt.get()
            num_word = que_c2s_rslt.get()
            if not args.without_ocr :
                recogRslt = que_c2s_rslt.get()
            if args.without_ocr or recogRslt != 'FAIL' :
                disp.display(img, inputImg, 1)
                disp.display(img, FuncAndClass.PILtoMat(imm1_pil), 2)
                disp.display(img, FuncAndClass.PILtoMat(imm2_pil), 3)
                disp.display(img, FuncAndClass.PILtoMat(imm3_pil), 4)
                disp.display(img, FuncAndClass.PILtoMat(imm4_pil), 5)
                disp.display(img, FuncAndClass.PILtoMat(licplate_pil), 6)
                window_idx = 7
                for char_pil in charsPIL_list :
                    if window_idx >= 15 :
                        break
                    disp.display(img, FuncAndClass.PILtoMat(char_pil), window_idx)
                    window_idx += 1
                if not args.without_ocr :
                    disp.paint(img, 15, (0,0,0))      
                    FuncAndClass.writePlate(img, recogRslt)
        if timeit.default_timer() - tstartMain > 1 :
            tstartMain = timeit.default_timer()
            img[152:172,1523:1820,:] = background[152:172,1523:1820,:]
            cv2.putText(img, str( datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') ), (1525,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,85), 2)
        cv2.imshow('img', img)
        if(cv2.waitKey(1) & 0xFF == ord('q')) :
            break
    # ----------------------------------------------------------------------------------------------------
    
    # exit  ----------------------------------------------------------------------------------------------
    print('[MAIN] terminate process')
    que_s2c.put('EXIT')                     # send signal to terminate REC process
    lock(key)                               # wait for REC process running out of all code
    clean(que_c2s_req)                      # clean buffer content so that REC process can close normally
    clean(que_c2s_rslt)
    clean(que_s2c)
    cv2.destroyAllWindows()
    cap.release()
    if(args.fps) :
        tendFps = timeit.default_timer()
        print( '[MAIN] FPS: {}'.format( frameCount/(tendFps-tstartFps) ) )
    # ----------------------------------------------------------------------------------------------------
