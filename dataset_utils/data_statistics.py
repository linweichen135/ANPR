from argparse import ArgumentParser
import csv

parser = ArgumentParser()
parser.add_argument('csv_path', help='the path of csv file')
args = parser.parse_args()

csvfile = open(args.csv_path, 'r')


num_number = []
num_alphabet = []
for i in range(0,10) :
    num_number.append(0)
for i in range(0,26) :
    num_alphabet.append(0)
    

numline = 0
line = csvfile.readline()
while True :
    line = csvfile.readline()
    numline += 1
    if not line :
        break
    imageName = ''
    islabel = False
    for char in line :
        if not islabel :
            if char == ',' :
                islabel = True
            continue
        ascii_char = ord(char)
        ascii_char -= 48
        if ascii_char >= 0 and ascii_char < 10 :
            num_number[ascii_char] += 1
            break
        ascii_char -= 17
        if ascii_char >= 0 and ascii_char < 26 and ascii_char != 8 and ascii_char != 14 :
            num_alphabet[ascii_char] += 1
            break
        print('Data error detected! Unrecognizable ' + char + ' in line ' + str(numline) + '.')
        exit()


for i in range(0,10) :
    print('Number of ' + chr(48+i) + ': ' + str(num_number[i]))
for i in range(0,26) :
    if i != 8 and i != 14 :
        print('Number of ' + chr(65+i) + ': ' + str(num_alphabet[i]))