import argparse
import json
from pathlib import Path
import cv2
from procces import procces
from sklearn.metrics import accuracy_score
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()
dir = Path(args.dir)

#bboxes=Path.joinpath(dir,'bboxes.txt')
bboxes=Path.joinpath(dir,'bboxes_gt.txt')

file = open(bboxes, 'r')
count = 0
FrameNames=[]
bboxesList=[]
gts=[]
# Strips the newline character
while True:
    FrameName=file.readline()
    if not FrameName:
        break
    FrameNames.append(str(FrameName)[:-1])
    N=file.readline()
    BoxesInFrame=[]
    # for i in range(int(N)):
    #     bbox=file.readline()[:-1]
    #     bbox=bbox.split(' ')
    #     bbox=[float(i) for i in bbox]
    #     BoxesInFrame.append(bbox)
    for i in range(int(N)):
        bbox=file.readline()[:-1]
        bbox=bbox.split(' ')
        gt,x,y,w,h=bbox
        
        gts.append(int(gt))
        bbox=[x,y,w,h]
        
        bbox=[float(i) for i in bbox]
        BoxesInFrame.append(bbox)
    bboxesList.append(BoxesInFrame)
file.close()
frameDir=Path.joinpath(dir,"frames")
frame2=cv2.imread(str(Path.joinpath(frameDir,FrameNames[0])),cv2.IMREAD_COLOR)
answers=[-1]
for i in range(int(len(FrameNames)-1)):
    frame1=frame2.copy()
    frame2=cv2.imread(str(Path.joinpath(frameDir,FrameNames[i+1])))
    #print(bboxesList[i+1])
    returnString=procces(frame1,frame2,bboxesList[i],bboxesList[i+1],0.4,1.0,0.0,0.0,0.0)#histWeight,TMWeight,IoUWeight
    # print(i)
    # print(bboxesList[i])
    # print(bboxesList[i+1])
    # print(returnString)
    answers.extend(returnString)
#print(gts)
# print(answers)
print(accuracy_score(gts,answers))