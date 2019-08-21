from argparse import ArgumentParser
from PIL import Image
import os
import csv
from datetime import datetime

parser = ArgumentParser()
parser.add_argument('srcData_directory', help='the path of directory placing images')
parser.add_argument('tgtData_directory', help='the path of directory to put renamed images')
parser.add_argument('--csv_path', help='the path of csv file')
parser.add_argument('--label', help='the label of images')
args = parser.parse_args()

writeCsv = False
if args.csv_path and args.label :
    try :
        csvfile = open(args.csv_path, 'a', newline='')
    except :
        print('Error when opening csv file.')
        exit()
    writer = csv.writer(csvfile)
    writeCsv = True
for root, dirs, files in os.walk( args.srcData_directory ) :
    for filename in files :
        img_pil = Image.open( args.srcData_directory + '/' + filename )
        output_filename = args.label + '_' + datetime.now().strftime('%Y%m%d%H%M%f') + '.png'
        img_pil.save( args.tgtData_directory + '/' + output_filename, 'PNG' )
        if writeCsv :
            writer.writerow( [output_filename , args.label] )
        output_filename += 1
        