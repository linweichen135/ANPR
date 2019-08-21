from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
base = BaseOverlay("base.bit")
import cv2

class camhdmi:
    def configure_camera (self, width, height, fps=30, buffer_size=0):
        self.camWidth = width
        self.camHeight = height
        self.camfps = fps
        self.buffer_size = buffer_size
    def configure_hdmi (self, width, height, fps=60):
        self.hdmiWidth = width
        self.hdmiHeight = height
        self.hdmifps = fps
    def start(self):
        # set and open camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camWidth)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camHeight)
        self.camera.set(cv2.CAP_PROP_FPS, self.camfps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        if not self.camera.isOpened():
            return False
        # set and open monitor
        mode = VideoMode(self.hdmiWidth, self.hdmiHeight, 24, self.hdmifps)
        self.hdmi_out = base.video.hdmi_out
        self.hdmi_out.configure(mode, PIXEL_BGR)
        self.hdmi_out.start()
        return True
    def readFrame(self):
        ret, frame = self.camera.read()
        if(ret):
            return frame
        else:
            raise RuntimeError("Failed to read from camera")
    def imshow(self, img):
        outframe = self.hdmi_out.newframe()
        outframe[0:self.hdmiHeight, 0:self.hdmiWidth, :] = img[0:self.hdmiHeight,0:self.hdmiWidth,:]
        self.hdmi_out.writeframe(outframe)
    def capture(self):
        for i in range(0,5):
            frame = self.readFrame()
        return frame
    def camCapShow(self):
        inframe = self.capture()
        if(self.camWidth != self.hdmiWidth or self.camHeight != self.hdmiHeight):
            inframe = cv2.resize(inframe, (self.hdmiWidth, self.hdmiHeight), interpolation=cv2.INTER_NEAREST)
        self.imshow(img)
    def free(self):
        self.camera.release()
        self.hdmi_out.close()
        del self.hdmi_out
