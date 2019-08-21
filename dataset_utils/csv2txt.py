from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('csv_path', help='the path of csv file')
parser.add_argument('txt1_path', help='the path of txt file recording image filenames')
parser.add_argument('txt2_path', help='the path of txt file recording image labels')
args = parser.parse_args()

csvfile = open(args.csv_path, 'r')
txt1 = open(args.txt1_path, 'w')
txt2 = open(args.txt2_path, 'w')


while True :
    line = csvfile.readline()
    if not line :
        break
    imageName = ''
    changeFile = False
    for char in line :
        if char == ',' :
            changeFile = True
            continue
        if not changeFile :
            imageName += char
        else :
            txt2.write(char)
    txt1.write(imageName + '\n')