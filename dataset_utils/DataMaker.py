from location import location
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import os
import PySimpleGUI as sg  
import win32gui
from datetime import datetime
 
 
def windowEnumerationHandler(hwnd, windows) :
    windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    
def setGUITopWindow() :
    windows = []
    win32gui.EnumWindows(windowEnumerationHandler, windows)
    for i in windows:
        if 'datamaker' in i[1].lower():
            win32gui.SetForegroundWindow(i[0])
            break
    

parser = ArgumentParser()
parser.add_argument('srcData_directory', help='the path of directory placing car images')
parser.add_argument('tgtData_directory', help='the path of directory to put output data')
parser.add_argument('label_filename', help='the filename of label file')
args = parser.parse_args()


with open(args.tgtData_directory + '/' + args.label_filename, 'a', newline='') as csvfile :
    layout = [[sg.Text('Input the label of current image')],      
              [sg.Input(key = 'label', do_not_clear=False)],
              [sg.Button('Read'), sg.Exit()]] 
    gui = sg.Window('DataMaker').Layout(layout) 
    plt.ion()
    writer = csv.writer(csvfile)  
    for root, dirs, files in os.walk( args.srcData_directory ) :
        for filename in files :
            img_pil = Image.open( args.srcData_directory + '/' + filename )
            _, _, _, _, success, num_word, charsPIL_list, _ = location(img_pil, blur=False)
            if success :
                for idx in range(num_word) :
                    char_pil = charsPIL_list[idx].copy()
                    plt.clf()
                    plt.imshow(char_pil)
                    values = {'label': None}
                    while not values['label'] :
                        setGUITopWindow()
                        event, values = gui.Read()
                        if event is None or event == 'Exit' :
                            exit()
                    if values['label'] == 'DD' :
                        print('Discarding image success.')
                    else :
                        output_filename = values['label'] + '_' + datetime.now().strftime('%Y%m%d%H%M%f') + '.png'
                        charsPIL_list[idx].save( args.tgtData_directory + '/' + output_filename, 'PNG' )
                        writer.writerow( [output_filename , values['label']] )
                        print(output_filename + ': ' + 'Labeling image success.' + ' (' + values['label'] + ')')
    gui.Close()