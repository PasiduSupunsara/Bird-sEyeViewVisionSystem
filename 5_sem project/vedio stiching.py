import cv2
import numpy as np

cap1 = cv2.VideoCapture("bike_left_01.mp4")
cap2 = cv2.VideoCapture("bike_right_01.mp4")

res1, frame1 = cap1.read()
res2, frame2 = cap2.read()

while res1 or res2:
    res1, frame1 = cap1.read()
    res2, frame2 = cap2.read()

    image_paths=[frame1,frame2]
    imgs = [frame1,frame2]

    imgs[0] = cv2.resize(image_paths[0],(0,0),fx=1,fy=1)
    imgs[1] = cv2.resize(image_paths[1],(0,0),fx=1,fy=1)
    stitchy=cv2.Stitcher.create()
    (dummy,output)=stitchy.stitch(imgs)


    # final output
    
    cv2.imshow('final result',output)
    
    cv2.imshow('img1',imgs[0])
    cv2.imshow('img2',imgs[1])

    

    
    if cv2.waitKey(25)  & 0xFF == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
