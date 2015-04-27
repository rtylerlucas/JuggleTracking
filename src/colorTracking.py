import cv2
import numpy as np
#Russell Lucas
cap = cv2.VideoCapture(0)
count = 0
coordinates = []
centroids = [(0,0), (0,0), (0,0)]
cx = 0
cy = 0
def centroidCalc():
    x1avg = 0
    y1avg = 0
    x2avg = 0
    y2avg = 0
    x3avg = 0
    y3avg = 0

    for x in range (0,(len(coordinates)/3)-1):
        x1avg = x1avg + coordinates[x][0]
        y1avg = y1avg + coordinates[x][1]
    x1avg = x1avg / (len(coordinates)/3)  
    y1avg = y1avg / (len(coordinates)/3)    
    centroids[0] = (x1avg, y1avg) 
    
    for y in range (( len(coordinates)/3), (len(coordinates) - (len(coordinates)/3)-1) ):
        x2avg = x2avg + coordinates[y][0]
        y2avg = y2avg + coordinates[y][1]
    x2avg = x2avg / (len(coordinates)/3)  
    y2avg = y2avg / (len(coordinates)/3)     
    centroids[1] = (x2avg, y2avg) 

    for z in range ( (len(coordinates) - (len(coordinates)/3)) , (len(coordinates)-1) ):
        x3avg = x3avg + coordinates[z][0]
        y3avg = y3avg + coordinates[z][1]
    x3avg = x3avg / (len(coordinates)/3)  
    y3avg = y3avg / (len(coordinates)/3)     
    centroids[2] = (x3avg, y3avg)    

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)


    # Threshold the HSV image to get only specific colors
    # define range of blue color in HSV
    # lower_blue = np.array([110, 50, 50], dtype=np.uint8)
    # upper_blue = np.array([130,255,255], dtype=np.uint8)
    # lower_pink = np.array([0, 51, 191], dtype=np.uint8)
    # upper_pink = np.array([5,160,255], dtype=np.uint8)
    # lower_orange = np.array([10, 50, 50], dtype=np.uint8)
    # upper_orange = np.array([30,255,255], dtype=np.uint8)
    lower_neon = np.array([27, 75, 100], dtype=np.uint8)
    upper_neon = np.array([47, 255, 255], dtype=np.uint8)
    # lower_white = np.array([0, 0, 200], dtype=np.uint8)
    # upper_white = np.array([255,20,255], dtype=np.uint8)
    upper_color = upper_neon
    lower_color = lower_neon
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing = cv2.dilate(closing,kernel, iterations = 2)
    closing = cv2.erode(closing,kernel, iterations = 4)

    contours, hierarchy = cv2.findContours(closing,1,2)
   
    try:
        (x,y),radius = cv2.minEnclosingCircle(contours[0])
        center = (int(x),int(y))
        radius = int(radius)
        cx = x
        cy = y
        cv2.circle(frame,center,radius,(0,255,0),2)
        coordinates.append(center)
        if(len(coordinates) > 30):
            coordinates.pop(0)
            centroidCalc()
        
    except IndexError:
        pass    

    cv2.circle(frame,centroids[0],2,(255,0,0),2)
    cv2.circle(frame,centroids[1],2,(0,255,0),2)
    cv2.circle(frame,centroids[2],2,(0,0,255),2)
    if( (( centroids[0][1] - centroids[1][1] ) > 50) and (( centroids[1][1] - centroids[2][1] ) > 50)):
        count = count + 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,str(count),(20,440), font, 2,(0,255,0))
    cv2.putText(frame,'x: ' + str(x) + ', y: ' + str(y),(300,30), font, 0.5,(0,0,255))
   
    cv2.imshow('frame',frame)

    cv2.imshow('mask',mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('z'):
        count = 0

cv2.destroyAllWindows()
