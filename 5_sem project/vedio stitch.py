import cv2
import numpy as np
import imutils

def rescale_frame(frame, percent = 75):
    scale_percent = 75
    width = int(frame.shape[1]*scale_percent/100)
    height = int(frame.shape[0]*scale_percent/100)
    dim = (width,height)
    return cv2.resize(frame,dim, interpolation= cv2.INTER_AREA)

clip1 = cv2.VideoCapture('bike_left_01.mp4')
clip2 = cv2.VideoCapture('bike_right_01.mp4')

width  = clip1.get(cv2.CAP_PROP_FRAME_WIDTH)
height = clip1.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps    = clip1.get(cv2.CAP_PROP_FPS) 
print('fps:', fps)              

ret1, frame1 = clip1.read()
ret2, frame2 = clip2.read()        

while True:

    ret1, frame1 = clip1.read()
    ret2, frame2 = clip2.read()    
    images = [frame1,frame2]
    
    
    imageStitcher = cv2.Stitcher_create()

    error, stitched_img = imageStitcher.stitch(images)

    if not error:
        stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))

        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        mask = np.zeros(thresh_img.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(areaOI)
        cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
        #cv2.rectangle(mask, (0,0), (20, 20), 255, -1)
        minRectangle = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            minRectangle = cv2.erode(minRectangle, None)
            sub = cv2.subtract(minRectangle, thresh_img)

        contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(areaOI)

        stitched_img = stitched_img[y:y + h, x:x + w]
       # stitched_img = rescale_frame(stitched_img, percent=30)
        cv2.imshow("Stitched Image Processed", cv2.resize(stitched_img,(500,300)))
        if cv2.waitKey(25)  & 0xFF == 27:
            break

    else:
        print("Images could not be stitched!")
        print("Likely not enough keypoints being detected!")