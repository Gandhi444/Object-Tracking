import argparse
from pathlib import Path
import cv2
from procces import procces

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()
dir = Path(args.dir)

bboxes=Path.joinpath(dir,'bboxes.txt')

file = open(bboxes, 'r')
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
    for i in range(int(N)):
        bbox=file.readline()[:-1]
        bbox=bbox.split(' ')
        x,y,w,h=bbox
        bbox=[x,y,w,h]
        bbox=[float(i) for i in bbox]
        BoxesInFrame.append(bbox)
    bboxesList.append(BoxesInFrame)
file.close()
frameDir=Path.joinpath(dir,"frames")
frame2=cv2.imread(str(Path.joinpath(frameDir,FrameNames[0])),cv2.IMREAD_COLOR)
answers=[[-1]*len(bboxesList[0])]
for i in range(int(len(FrameNames)-1)):
    frame1=frame2.copy()
    frame2=cv2.imread(str(Path.joinpath(frameDir,FrameNames[i+1])))
    returnString=procces(frame1,frame2,bboxesList[i],bboxesList[i+1],newTH=0.5,histWeight=1.08,TMWeight=1.3,
                         IoUWeight=0.05,SizeWeight=0.65,SSIMWeight=1.16)
    answers.append(returnString)
for frame_answer in answers:
    string=""
    for number in frame_answer:
        string=string + " " + str(number)
    print(string)