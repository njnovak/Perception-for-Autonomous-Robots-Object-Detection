import cv2
from numpy import linalg
import numpy as np
import math

video_source = cv2.VideoCapture('ball.mov')
if video_source.isOpened() == False:
    print("Error opening video")
reading, frame = video_source.read()
xf,yf,__ = frame.shape

# xf = int(round(xf*0.6))
# yf = int(round(yf*0.6))
out = cv2.VideoWriter('ballDetected.mov',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (yf,xf))
none_count = 0
while reading:
    reading, frame = video_source.read()

    if reading:
        
        shapes = frame
        
        shapes_grayscale = cv2.cvtColor(shapes, cv2.COLOR_RGB2GRAY)
        shapes_grayscale = cv2.bitwise_not(shapes_grayscale)

        shapes_blurred = cv2.medianBlur(shapes_grayscale, 5)
        rows = shapes_blurred.shape[0]

        # Get Hough Circle
        circle_vals = cv2.HoughCircles(shapes_blurred, cv2.HOUGH_GRADIENT, 1, 10, param1=60, param2=27,minRadius=0, maxRadius=0)
        if circle_vals is not None:
            # print(circle_vals)
            circles = np.uint16(np.around(circle_vals))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(frame,center,1,(100,100,100), 2)

                # rad = i[2]
                rad = 11
                cv2.circle(frame, center, rad, (255, 0, 255), 2)

        else:
            none_count += 1
        out.write(shapes)
        # Uncomment to view each frame result : %--------
        # cv2.imshow("Results", frame)
        
        # key = cv2.waitKey(100) #pauses for .1 seconds before fetching next image
        # if key == 27: #if ESC is pressed, exit loop and kill program
        #     cv2.destroyAllWindows()
        #     break
        # ---------%
    else:
        break

# print("Frames Missed: ",none_count)
out.release()
cv2.destroyAllWindows()
