from argparse import ArgumentParser
from datetime import datetime
import csv
import cv2
import os

parser = ArgumentParser()
parser.add_argument('srcData_directory', help='the path to the original dataset directory')
parser.add_argument('tgtData_directory', help='the path to the augmented dataset directory')
parser.add_argument('label_filename', help='the filename of label file')
args = parser.parse_args()

with open(args.tgtData_directory + '/' + args.label_filename, 'a', newline='') as csvfile :
    writer = csv.writer(csvfile)
    for root, dirs, files in os.walk( args.srcData_directory ) :
        for filename in files :
            if filename == args.label_filename :
                continue
            print('Processing ', filename, ' ...')
            img = cv2.imread( args.srcData_directory + '/' + filename )
            img_blur1 = cv2.blur(img, (3,3))
            img_blur2 = cv2.blur(img, (5,5))
            label = filename[0]
            outputfile1 = label + '_' + datetime.now().strftime('%Y%m%d%H%M%f') + '.png'
            outputfile2 = label + '_' + datetime.now().strftime('%Y%m%d%H%M%f') + '.png'
            cv2.imwrite(args.tgtData_directory + '/' + outputfile1, img_blur1)
            cv2.imwrite(args.tgtData_directory + '/' + outputfile2, img_blur2)
            writer.writerow( [outputfile1 , label] )
            writer.writerow( [outputfile2 , label] )