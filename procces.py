import cv2 as cv,cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
def procces(frame1:cv2.Mat,frame2:cv2.Mat,bboxs1,bboxs2,newTH,histWeight,TMWeight,IoUWeight):
    m=len(bboxs1)
    #print(m)
    n=len(bboxs2)
    #print(m,n)
    graph=np.zeros((m+n,n))
    graph[m:,:]=newTH
    cutOutBoxes1=[]
    for bbox in bboxs1:
        x,y,w,h = [int(i) for i in bbox]
        cutOutBoxes1.append(frame1.copy()[y:y+h,x:x+w])
    cutOutBoxes2=[]
    for bbox in bboxs2:
        x,y,w,h = [int(i) for i in bbox]
        cutOutBoxes2.append(frame2.copy()[y:y+h,x:x+w])
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
            # Intersection height and width.
            i_height = np.maximum(y2 - y1 + 1, np.array(0.))
            i_width = np.maximum(x2 - x1 + 1, np.array(0.))
            
            area_of_intersection = i_height * i_width
            
            
            area_of_union = box1[2] * box1[3] + box2[2] * box2[3] - area_of_intersection
            
            iou = area_of_intersection / area_of_union

            IoUMatrix[i,j]=iou
    TMMatrix=np.zeros((m,n))
    for i in range(m):
        #print("###############################################")
        for j in range(n):
            shape1=cutOutBoxes1[i].shape
            shape2=cutOutBoxes2[j].shape
            width=min(shape1[1],shape2[1])
            height=min(shape1[0],shape2[0])
            input1 = cv2.resize(cutOutBoxes1[i], (width,height), interpolation = cv2.INTER_AREA)
            input2 = cv2.resize(cutOutBoxes2[j], (width,height), interpolation = cv2.INTER_AREA)
            res = cv2.matchTemplate(input1,input2,cv2.TM_CCORR_NORMED)
            # print(res)
            # cv2.imshow("in1",input1)
            # cv2.imshow("in2",input2)
            # cv.waitKey(0)
            TMMatrix[i,j]=res[0][0]
    #print(TMMatrix)
    HistMatrix=np.zeros((m,n))
    for i in range(m):
        #print("################################")
        for j in range(n):
            #channels = [1]
            r1= cv2.calcHist([cutOutBoxes1[i]], [0], None, [256], [0,256])
            cv.normalize(r1, r1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            r2= cv2.calcHist([cutOutBoxes2[j]], [0], None, [256], [0,256])
            cv.normalize(r2, r2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            g1= cv2.calcHist([cutOutBoxes1[i]], [1], None, [256], [0,256])
            cv.normalize(g1, g1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            g2= cv2.calcHist([cutOutBoxes2[j]], [1], None, [256], [0,256])
            cv.normalize(g2, g2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            b1= cv2.calcHist([cutOutBoxes1[i]], [2], None, [256], [0,256])
            cv.normalize(b1, b1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            b2= cv2.calcHist([cutOutBoxes2[j]], [2], None, [256], [0,256])
            cv.normalize(b2, b2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            method=cv2.HISTCMP_CORREL
            compR=cv2.compareHist(r1,r2,method)
            compG=cv2.compareHist(g1,g2,method)
            compB=cv2.compareHist(b1,b2,method)
            score=(compB+compR+compG)/3
            score=max(0,min(score,1))

            # print(score)
            # cv2.imshow("in1",cutOutBoxes1[i])
            # cv2.imshow("in2",cutOutBoxes2[j])
            # cv.waitKey(0)
            HistMatrix[i,j]=score
            
    SimilarityMatrix=(HistMatrix*histWeight+IoUMatrix*IoUWeight+TMMatrix*TMWeight)/(TMWeight+IoUWeight+histWeight)
    #print(TMMatrix)
    graph[:m,:n]=SimilarityMatrix
    #print(1-graph)
    row,col=linear_sum_assignment(1-graph)
    col[col>m-1]=-1
    #print(col)
    return col