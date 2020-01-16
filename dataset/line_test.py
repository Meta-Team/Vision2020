# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
'''本文件仅作临时测试用，无功能'''

def im_show(path):
    img = cv2.imread(path)
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edge = cv2.Canny(imgray, 50, 180, apertureSize=3)
    # plt.imshow(edge)
    # plt.show()
    # Hough直线检测
    # height = int(img.shape[0]*0.8)
    # lines = cv2.HoughLinesP(edge, 1, np.pi/180,20)


    lines =[]
    # min_length = int(img.shape[0]*0.8)
    # _, imred = cv2.threshold(imred, 50, 255, cv2.THRESH_OTSU)
    # lines.extend(cv2.HoughLinesP(imred, 1, np.pi/180, min_length))

    # imblue = img[:,:,2]
    # _, imblue = cv2.threshold(imblue, 50, 255, cv2.THRESH_OTSU)
    # lines.extend(cv2.HoughLinesP(imblue, 1, np.pi/180, min_length))

    min_length = int(img.shape[0]*0.8)
    print(min_length)

    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img_=cv2.threshold(imgray,175,255,cv2.THRESH_BINARY_INV)

    _, imred = cv2.threshold(img[:,:,2], 100, 255, cv2.THRESH_OTSU)
    imred = cv2.bitwise_and(imred,img_)
    lines = cv2.HoughLinesP(imred, 1, np.pi/180, min_length)

    if lines is None:
        print('error')
        return
    max_length = min_length**2
    max_line = (0,0,0,0)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if (x1-x2)**2+(y1-y2)**2 > max_length:
            max_length = (x1-x2)**2+(y1-y2)**2
            max_line = line[0]

    x1,y1,x2,y2 = max_line
    cv2.line(img, (x1,y1), (x2,y2),(0,255,0),1)
    plt.imshow(img)
    plt.show()


#%%

path = '/mnt/e/robomaster/mydump/robomaster_South China Regional Competition/lightbar/'
for x in os.listdir(path):
    filepath = os.path.join(path,x)
    im_show(filepath)
# %%

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

# %%


path = '/mnt/e/robomaster/mydump/robomaster_South China Regional Competition/lightbar/EvolutionVs华南虎_BO3_2_216_13.jpg'
img = cv2.imread(path)
plt.imshow(img)
plt.show()

min_length = int(img.shape[0]*0.8)

imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,img_=cv2.threshold(imgray,150,255,cv2.THRESH_BINARY_INV)
plt.imshow(img_)
plt.show()

_, imred = cv2.threshold(img[:,:,0], 125, 255, cv2.THRESH_BINARY)
imred = cv2.bitwise_and(imred,img_)

plt.imshow(imred)
plt.show()
lines = cv2.HoughLinesP(imred, 1, np.pi/180, min_length)

# %%
