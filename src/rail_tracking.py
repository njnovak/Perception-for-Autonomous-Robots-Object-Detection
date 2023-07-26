import cv2
from numpy import linalg
import numpy as np
import math
import matplotlib.pyplot as plt
# Nicholas Novak
# ENPM 673 Spring 2023

def findHMat(src, dest, N):
    '''
    Ported over from project 2, where this was originally used. The same method applies here, though
    '''
    A = []
    for i in range(N):
        x, y = src[i][0], src[i][1]
        xp, yp = dest[i][0], dest[i][1]

        A.append([x, y, 1, 0, 0, 0, -x*xp, -xp*y, -xp])
        A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H


img = cv2.imread('train_track.jpg')
# print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
dst = cv2.Canny(gray, 900, 1200, None, 3)
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

line_equations = []
linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 150, 400, 100, 10)
    
if linesP is not None:
    for i in range(0, 2):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        pt1 = (l[0], l[1])
        pt2 = (l[2], l[3])
        m = (l[1] - l[3]) / (l[0]-l[2])
        b = pt2[1] - (m*pt2[0])
        
        # Now we find the lower x intercepts of the lines at max y:
        h_max = cdst.shape[0] # height of the image (2k)
        x_int = round((h_max - b)/m)

        x2_int = round(((h_max/2) - b)/m)
        out_pt1 = (x_int, h_max)
        out_pt2 = (x2_int, round(h_max/2))
        cv2.line(cdstP, out_pt1, out_pt2, (0,0,255), 3, cv2.LINE_AA)

        line_equations.append({i:[m,b,x_int,out_pt1,out_pt2]})
    cv2.imwrite("linesOnRails.jpg", cdstP)


    # print(line_equations)
    # Now we have our 2 line equations and 4 points!
    rest_of_rail_num = -50

    p_1 = line_equations[0][0][3]
    p_2 = line_equations[0][0][4]
    p_3 = (line_equations[1][1][3][0],line_equations[1][1][3][1])
    p_4 = (line_equations[1][1][4][0],line_equations[1][1][4][1])
    src_pts = np.float32([p_1,p_2,p_3,p_4])

    ps_1 = [0,cdst.shape[0]]
    ps_2 = [0,0]
    ps_3 = [line_equations[1][1][3][0] - line_equations[0][0][3][0] + rest_of_rail_num,cdst.shape[0]] # Includes rest of the rail that is getting cut off
    ps_4 = [line_equations[1][1][3][0] - line_equations[0][0][3][0] + rest_of_rail_num,0]


    new_shape = (line_equations[1][1][3][0] - line_equations[0][0][3][0],round(cdst.shape[0]))
    dest_pts = np.float32([ps_1,ps_2,ps_3,ps_4])
    H = findHMat(src_pts,dest_pts,4)
    img_Mat = cv2.getPerspectiveTransform(src_pts,dest_pts)
    # print(img_Mat)

    img = cv2.warpPerspective(img, img_Mat, new_shape,flags=cv2.INTER_LINEAR)
    graywarp = cv2.warpPerspective(gray, img_Mat, new_shape,flags=cv2.INTER_LINEAR)

# print(src_pts)
# print(dest_pts)

#Uncomment to view warped image
# cv2.imshow("warped source", img)
cv2.imwrite("warpedRails.jpg", img)


# cv2.imshow("Detected Lines - Probabilistic Line Transform", cdstP)

# To find average distance, we want to go from the inner part of each tie to the other one
# mask = np.where()
thresh = 140
assignedVal = 255
thresh_method = cv2.THRESH_BINARY
_, dst2 = cv2.threshold(graywarp,thresh,assignedVal, thresh_method)
dst2[:,127:1125] = 0 # Remove center white pixels
# Uncomment to view warped mask
# cv2.imshow("line warp", dst2)

dist_list = []
for row in dst2:
    x1_pos = [0]
    x2_pos = [len(row)]
    for col in range(0,127): #First rail:
        if row[col] != 0:
            x1_pos.append(col)
    for col in range(1125,len(row)): #Second rail:
        if row[col] != 0:
            x2_pos.append(col)
    rail1_ave = sum(x1_pos) / len(x1_pos)
    rail2_ave = sum(x2_pos) / len(x2_pos)
    dist_list.append(rail2_ave - rail1_ave)
plt.xlabel("Row number")
plt.ylabel("Distance between rail centers (pixels)")
plt.grid()
plt.title("Pixel distance of warped rails in each row")
x = list(range(len(dist_list)))
plt.plot(x,dist_list)
plt.savefig('DistancePlot.jpg')
average_dist = sum(dist_list) / len(dist_list)
print("Average distance: ", average_dist, " pixels")
# Uncomment to prevent images disappearing
# cv2.waitKey()
# plt.show()
cv2.destroyAllWindows()


