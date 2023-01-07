import cv2
import numpy as np

clip1 = cv2.VideoCapture('left.mp4')
clip2 = cv2.VideoCapture('right.mp4')

width  = clip1.get(cv2.CAP_PROP_FRAME_WIDTH)
height = clip1.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps    = clip1.get(cv2.CAP_PROP_FPS) 
print('fps:', fps)              

video_format = cv2.VideoWriter_fourcc(*'MP42')  # .avi 
final_clip   = cv2.VideoWriter('output.avi', video_format, fps, (int(width), int(height)))

delay = int(1000/fps)
print('delay:', delay)    
          

while True:

    ret1, frame1 = clip1.read()
    ret2, frame2 = clip2.read()    

    if not ret1 or not ret2:
        break

          
    final_frame = np.vstack([frame1, frame2])  # two videos in one row
    final_clip.write(final_frame)

    cv2.imshow('Video', final_frame)
    #cv2.imshow("first",frame1)
    #cv2.imshow("second",frame2)
    key = cv2.waitKey(delay) & 0xFF
    
    if key == 27:
        break
    
cv2.destroyWindow('Video')

clip1.release()
clip2.release()