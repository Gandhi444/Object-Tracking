import cv2 as cv,cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity
def procces(frame1:cv2.Mat,frame2:cv2.Mat,bboxs1,bboxs2,newTH,histWeight,TMWeight,IoUWeight,SizeWeight,SSIMWeight):
    m=len(bboxs1)
    n=len(bboxs2)
    graph=np.zeros((m+n,n))
    graph[m:,:]=newTH
    cutOutBoxes1=[]
    fixedBoxes1=[]
    for bbox in bboxs1:
        x,y,w,h = [int(i) for i in bbox]
        x=max(x,0)
        y=max(y,0)
        w=min(x+w,frame1.shape[1])-x
        h=min(y+h,frame1.shape[0])-y
        fixedBoxes1.append((x,y,w,h))
        cutOutBoxes1.append(frame1.copy()[y:y+h,x:x+w])

    cutOutBoxes2=[]
    fixedBoxes2=[]
    for bbox in bboxs2:
        x,y,w,h = [int(i) for i in bbox]
        x=max(x,0)
        y=max(y,0)
        w=min(x+w,frame2.shape[1])-x
        h=min(y+h,frame2.shape[0])-y
        fixedBoxes2.append((x,y,w,h))
        cutOutBoxes2.append(frame2.copy()[y:y+h,x:x+w])
    bboxs1=fixedBoxes1
    bboxs2=fixedBoxes2

    IoUMatrix=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            box1=bboxs1[i]
            box2=bboxs2[j]
            x1 = np.maximum(box1[0], box2[0])
            y1 = np.maximum(box1[1], box2[1])
            x2 = np.minimum(box1[0]+box1[2], box2[0]+box2[2])
            y2 = np.minimum(box1[1]+box1[3], box2[1]+box2[3])
            x,y,w,h = [int(i) for i in box1]
            input1=cv2.rectangle(frame1.copy(), (x, y), (x + w, y + h), (255,0,0), 4)
            x,y,w,h = [int(i) for i in box2]
            input2=cv2.rectangle(frame2.copy(), (x, y), (x + w, y + h), (0,255,0), 4)
            i_height = np.maximum(y2 - y1 + 1, np.array(0.))
            i_width = np.maximum(x2 - x1 + 1, np.array(0.))
            
            area_of_intersection = i_height * i_width
            
            
            area_of_union = box1[2] * box1[3] + box2[2] * box2[3] - area_of_intersection
            
            iou = area_of_intersection / area_of_union
            IoUMatrix[i,j]=iou


    TMMatrix=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            shape1=cutOutBoxes1[i].shape
            shape2=cutOutBoxes2[j].shape
            width=min(shape1[1],shape2[1])
            height=min(shape1[0],shape2[0])
            input1 = cv2.resize(cutOutBoxes1[i], (width,height), interpolation = cv2.INTER_AREA)
            input2 = cv2.resize(cutOutBoxes2[j], (width,height), interpolation = cv2.INTER_AREA)
            res = cv2.matchTemplate(input1,input2,cv2.TM_CCORR_NORMED)
            TMMatrix[i,j]=res[0][0]


    HistMatrix=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            img1=cv2.cvtColor(cutOutBoxes1[i],cv2.COLOR_BGR2HSV)
            img2=cv2.cvtColor(cutOutBoxes2[j],cv2.COLOR_BGR2HSV)
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges
            h_bins = 50
            s_bins = 60
            histSize = [h_bins, s_bins]
            hist1= cv.calcHist([img1], [0, 1], None, histSize, ranges, accumulate=False)
            hist1=cv.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            hist2= cv.calcHist([img2], [0, 1], None, histSize, ranges, accumulate=False)
            hist2=cv.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            comp=cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL)
            HistMatrix[i,j]=comp
    SizeMatrix=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            area1=bboxs1[i][2]*bboxs1[i][3]
            area2=bboxs2[j][2]*bboxs2[j][3]
            ratio=min(area1,area2)/max(area1,area2)
            SizeMatrix[i][j]=ratio
    SSIMMatrix=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            shape1=cutOutBoxes1[i].shape
            shape2=cutOutBoxes2[j].shape
            width=min(shape1[1],shape2[1])
            height=min(shape1[0],shape2[0])
            input1 = cv2.resize(cutOutBoxes1[i], (width,height), interpolation = cv2.INTER_AREA)
            input2 = cv2.resize(cutOutBoxes2[j], (width,height), interpolation = cv2.INTER_AREA)
            input1=cv2.cvtColor(input1,cv.COLOR_BGR2GRAY)
            input2=cv2.cvtColor(input2,cv.COLOR_BGR2GRAY)
            score = structural_similarity(input1, input2)
            SSIMMatrix[i][j]=score


    SimilarityMatrix=HistMatrix*histWeight+IoUMatrix*IoUWeight+TMMatrix*TMWeight+SizeMatrix*SizeWeight+SSIMMatrix*SSIMWeight
    WeightSum=TMWeight+IoUWeight+histWeight+SizeWeight+SSIMWeight
    SimilarityMatrix=SimilarityMatrix/WeightSum
    graph[:m,:n]=SimilarityMatrix

    row,col=linear_sum_assignment(graph,maximize=True)
    pairs=[]
    for i in range(len(row)):
        pairs.append((col[i],row[i]))
    sorted_pairs=sorted(pairs)
    out=[]
    for pair in sorted_pairs:
        out.append(pair[1])
    out=np.array(out)
    out[out>m-1]=-1
    return out