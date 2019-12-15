import numpy as np
import cv2
def computeIOU( rec1, rec2 ):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
    return (intersect / (sum_area - intersect))*1.0


def near( box1, box2 ):
    '''
    determine whether two boxes is nearly the same
    '''
    box1=(box1[0],box1[1],box1[0]+box1[2],box1[1]+box1[3])
    box2=(box2[0],box2[1],box2[0]+box2[2],box2[1]+box2[3])
    iou = computeIOU(box1,box2)
    #print(box1, box2,iou)
    return iou > 0.8


def maskHSV( img,TH_V ):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, TH_V), (255, 255, 255))
    return mask

def classify( img, TH_V=240, TH_SIZE=36, TH_WHITE=15 ):
    mask = maskHSV( img,TH_V )
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    num = 0
    x,y = 0,0
    sum = np.array([0,0,0])
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            if mask[ i,j ]:
                sum += lab[ i,j ]
                x += i
                y += j
                num += 1
    if num <= TH_SIZE:
        return 'none', -1, -1
    sum = sum / num
    x = int(x/num)
    y = int(y/num)
    blue = np.linalg.norm(sum-np.array([188, 146,  87]))
    red = np.linalg.norm(sum-np.array([200, 158, 140]))
    white = np.linalg.norm(sum-np.array([255, 128, 128]))
    #print('blue red white\n',blue,red,white)
    #print(num)
    if white <= TH_WHITE:
        return 'none',x,y
    elif red < blue:
        return 'red',x,y
    else:
        return 'blue',x,y

def findLight( image ):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mser = cv2.MSER_create(_max_area = 1000, _min_area=50)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(gray)
    boxes = list(boxes)
    num = 0
    tmp = []
    boxes.sort(key=lambda x:(x[0], x[1]))
    for idx,box in enumerate(boxes):
        x, y, w, h = box
        if w >= h:
            continue
        #if h >= 2*w:
        #    continue
        if idx != 0:
            if(near(boxes[idx], boxes[idx-1])):
                continue
        num += 1
        #cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0), 2)
        tmp.append(box)

    boxes = tmp
    #print(len(boxes))
    for box in boxes:
        x, y, w, h = box
        x1 = x
        y1 = y
        x2 = x1+w
        y2 = y1+h
        img2 = np.copy(img[y1:y2,x1:x2,:])
        color, y, x = classify(img2)
        if color == 'red':
            cv2.circle(image, (x1+x,y1+y), 5, (0,0,0), thickness=-1)
        if color == 'blue':
            cv2.circle(image, (x1+x,y1+y), 5, (0,0,255), thickness=-1)
    return image

if __name__ == '__main__':
    video=cv2.VideoCapture(0)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), \
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    while True:
        ret,img=video.read()
        img = findLight(img)
        if ret is False:
            exit()
        cv2.namedWindow('video',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF ==27:
            video.release()
            break