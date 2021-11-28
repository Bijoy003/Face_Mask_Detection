import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


# image to npy
DIRECTORY = r"./images/"
CATEGORIES = ["with_mask", "without_mask","incorrect_mask"]


for category in CATEGORIES:
    data=[]
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        #cv2.imshow('images',image)
        image = cv2.imread(img_path)
        image=cv2.resize(image,(100,100))
        data.append(image)
        cv2.imshow('image', image)
        print(len(data))
        if cv2.waitKey(2)==27:
            break
    np.save(category + '.npy', data)
#print(data[0])
plt.imshow(data[0])
print('image to .npy done')
