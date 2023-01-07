import cv2
import numpy as np
import time

cap = cv2.VideoCapture("myvedio.mp4")
res, frame = cap.read()
time_pre = time.time()
while res:
    
    res, frame = cap.read()
    pts1 = np.float32([[0, 260], [640, 260],
                       [0, 400], [640, 400]])
    pts2 = np.float32([[0, 0], [400, 0],
                       [0, 640], [400, 640]])
     

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (640, 480))
     
    cv2.imshow('frame', frame)
    cv2.imshow('frame1', result)
    time_cur = time.time() 
    print("fps = " , 1/(time_cur - time_pre))
    if cv2.waitKey(1)== 27:
        break

cap.release()
cv2.destroyAllWindows()