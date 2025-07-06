import cv2
import numpy as np
from PIL import Image

def similar(img1, img2, threshold=0.4):
    img1 = img1.resize((100, 100)).convert("RGB")
    img2 = img2.resize((100, 100)).convert("RGB")

    h1 = cv2.calcHist([np.array(img1)], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    h2 = cv2.calcHist([np.array(img2)], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)

    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()

    similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return similarity > threshold
