import cv2 as cv

cap = cv.VideoCapture("in.avi")
ret, frame1 = cap.read()
ret, frame2 = cap.read()
while cap.isOpened():
    diff = cv.absdiff(frame1,frame2)
    diff_gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(diff_gray,(5,5),0)
    _, thresh = cv.threshold(blur,20,255,cv.THRESH_BINARY)
    dilated = cv.dilate(thresh,None,iterations=3)
    contours,_ = cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x,y,w,h) = cv.boundingRect(contour)
        if cv.contourArea(contour) < 900:
            continue
        cv.rectangle(frame1,(x,y),(x+w,y+h),(0,155,0),2)
        cv.putText(frame1,"Human Detection",(10,20),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

    cv.imshow("video",frame1)
    frame1 = frame2
    ret,frame2 = cap.read()

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()