import cv2
import numpy as np

#Web camera
cap = cv2.VideoCapture("video.mp4")
min_width_react = 80 #min width rectangle
min_height_react = 80 #min height rectangle
count_line_position = 550  # Define the position of the counting line

#Initialize the background subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()  # Create a background subtractor objectd:\Downloads\videoplayback.mp4
def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy

detect = []
offset = 6 #Allowable error between pixel
counter = 0

while True:
    ret, frame1 = cap.read()  # Read a frame from the webcam
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    blur = cv2.GaussianBlur(grey, (3,3),5)
   
    # Applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # Dilate the image to fill in holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # Create a kernel for morphological operations
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Apply morphological closing to the dilated image
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)  # Apply morphological closing again to the image
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the processed image
    # cv.RETR_LIST − Retrieves all of the contours without establishing any hierarchical relationships.
    # cv.RETR_TREE − Retrieves all of the contours and reconstructs a full hierarchy of nested contours.
    # cv.CHAIN_APPROX_SIMPLE − Compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)  # Draw the counting line on the frame
    
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c) # boundingRect() function in OpenCV calculates the smallest upright (axis-aligned) rectangle that completely encloses a contour (or a set of points).
        validate_counter = (w >= min_width_react and h >= min_height_react)
        if not validate_counter:
            continue
        cv2.rectangle(frame1,(x,y), (x+w,y+h),(0,255,0),2)   
        cv2.putText(frame1, "Vehicle"+str(counter), (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,244,0),2)
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0,0,255),-1)
        
        for (x,y) in detect:
            if y<(count_line_position + offset) and y>(count_line_position - offset):
                counter+=1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                detect.remove((x,y))
                print("Vehice Counter:"+str(counter))
    cv2.putText(frame1, "VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    
    #cv2.imshow("Detector", dilatada)  # Display the processed image 
    cv2.imshow("Video Original", frame1)  # Display the webcam feed 
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()