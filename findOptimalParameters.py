import argparse
import json
from pathlib import Path
import cv2
from procces import procces
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize,dual_annealing
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
frames=[]
for i in range(int(len(FrameNames))):
    frames.append(cv2.imread(str(Path.joinpath(frameDir,FrameNames[i]))))
def fun(x):
    answers=[-1]
    frame2=cv2.imread(str(Path.joinpath(frameDir,FrameNames[0])),cv2.IMREAD_COLOR)
    for i in range(int(len(FrameNames)-1)):
        frame1=frames[i]
        frame2=frames[i+1]
        returnString=procces(frame1,frame2,bboxesList[i],bboxesList[i+1],x[0],x[1],x[2],x[3],x[4],x[5])
        answers.extend(returnString)
    return 1-accuracy_score(gts,answers)
x0=[0.5,1,1,1,1]
bnds=((0.05,1),(0.05,10),(0.05,10),(0.05,10),(0.05,10))
#res=minimize(fun,x0, bounds=bnds)
res=dual_annealing(fun,bnds,maxiter=100)
print(res)




