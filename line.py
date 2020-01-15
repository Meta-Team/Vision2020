import cv2
import numpy as np
import matplotlib.pyplot as plt


path = '/mnt/e/robomaster/mydump/robomaster_Central China Regional Competition/image/AllianceVsArtisans_BO2_2_1_13.jpg'
img = cv2.imread(path)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(imgray, 50, 180, apertureSize=3)
plt.imshow(edge)
plt.show()
# Hough直线检测
height = int(img.shape[0]*0.8)
print(height)
lines = cv2.HoughLinesP(edge, 1, np.pi/180,20)
print(len(lines))
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img, (x1,y1), (x2,y2),(0,255,0),1)

img = img[:, :, ::-1]
plt.imshow(img)
plt.show()

# 显示所在直线完整直线
# lines = cv2.HoughLine(edge, 1, np.pi/180,20)
# for rho, theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*a)
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*a)

#     cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 1)