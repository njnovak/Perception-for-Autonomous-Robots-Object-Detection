import cv2
from numpy import linalg
import numpy as np
import math
import matplotlib.pyplot as plt
# Nicholas Novak
# ENPM 673 Spring 2023

img = cv2.imread('hotairbaloon.jpg')
gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

_, mask = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_not(mask)
mask[416:759,573:1080] = 0
mask[0:755,0:384] = 0


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask[2236:mask.shape[0],4337:mask.shape[1]] = 0
mask[2236:2284,4827:mask.shape[1]] = 0
# Uncomment to view mask:
# cv2.imshow("mask",mask)
# cv2.waitKey()
n_components, output, stats, centroids = cv2.connectedComponentsWithStats(
      mask, connectivity=8)
# print(stats)

for i in range(1, n_components): # Leave out first component since it is the entire frame
      x = stats[i, cv2.CC_STAT_LEFT]
      y = stats[i, cv2.CC_STAT_TOP]
      w = stats[i, cv2.CC_STAT_WIDTH]
      h = stats[i, cv2.CC_STAT_HEIGHT]
      area = stats[i, cv2.CC_STAT_AREA]
      (cX, cY) = centroids[i]

      color = list(np.random.random(size=3) * 256)
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
      cv2.circle(img, (int(cX), int(cY)), 4, (0, 0, 255), -1)
      cv2.putText(img, "Balloon(s) Number "+str(i), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
      for row in range(y,y+h):
        for col in range(x,x+w):
            if mask[row,col] == 255:
                img[row,col] = color
            

# print(n_components)

cv2.imwrite('DetectedBalloons.jpg',img)

# Uncomment to view results:
# cv2.imshow("Filtered Components", img)
# cv2.waitKey(0)    
cv2.destroyAllWindows()



