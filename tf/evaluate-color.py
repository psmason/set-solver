import os
import sys
import socket

import numpy as np
from scipy import ndimage
import scipy.misc
import cv2

raw = ndimage.imread(sys.argv[1])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
rc, data = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

sock.connect(("localhost", 9000))
sock.sendall(np.getbuffer(data))
received = sock.recv(1024)
sock.close()
print "Prediction: {}".format(received)
