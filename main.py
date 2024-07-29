import cv2
import cvzone
import cvzone.FPS
from cvzone.SelfiSegmentationModule import SelfiSegmentation as selfSeg
import os

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor=selfSeg()
#fpsReader=cvzone.FPS()

#imgBg=cv2.imread("images/img1.jpg")
#imgBg = cv2.resize(imgBg, (640, 480))

ImgList=os.listdir("images")
lst=[]
for i in ImgList:
    ok=cv2.imread(f'images/{i}')
    ok = cv2.resize(ok, (640, 480))
    lst.append(ok)
#print(len(lst))  

idx=0

while True:
    success,img=cap.read()
    imgOut=segmentor.removeBG(img,lst[idx],cutThreshold=0.88)

    stackImg=cvzone.stackImages([img,imgOut],2,1)
    #_,stackImg=fpsReader.update(stackImg,color=(0,255,255))

    print(idx)
    cv2.imshow("Front Cam", stackImg)       
    key=cv2.waitKey(1) 
    if(key==ord('a')):
        if(idx>0):
            idx-=1
    elif(key==ord('d')):
        if(idx<len(lst)-1):
            idx+=1
    elif(key==ord('s')):
        break        