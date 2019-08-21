import cv2
import timeit
import sys
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('camera_number', type=int, help='the number of camera which is going to be opened')
parser.add_argument('frame_number', type=int, help='the number of frames to be tested')
args = parser.parse_args()


cap = cv2.VideoCapture(args.camera_number)
if(not cap.isOpened()):
    print('Failed to open camera.')
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


start = timeit.default_timer()
for i in range(args.frame_number):
    _, frame = cap.read()
end = timeit.default_timer()


print('FPS: {}'.format(args.frame_number/(end-start)))
cap.release()