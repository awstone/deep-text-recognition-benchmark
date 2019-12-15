import cv2
import glob
import os

for file in glob.iglob('./lineImages/**/**/*.tiff'):
    os.remove(file)

    