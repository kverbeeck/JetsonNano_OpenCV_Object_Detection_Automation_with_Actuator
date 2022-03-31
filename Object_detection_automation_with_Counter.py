import cv2
print (cv2.__version__)

################################################################
path = '/home/jetson/PycharmProjects/openCV-contrib-python/venv/haar-cascade-files-master/Cascade_Poesjes.xml'  # PATH OF THE CASCADE
objectName = 'Poesje'  # OBJECT NAME TO DISPLAY
frameWidth = 640  # DISPLAY WIDTH
frameHeight = 480  # DISPLAY HEIGHT
color = (255, 0, 255)
#################################################################


# Size of the camera Display Window (Width, Height, Rotation), values times 2 (2x) to make bigger
dispW = 640
dispH = 480
#Rotate the camera output 0°
flip = 0
# nvarguscamerasrc: Launches GStreamer that runs the raspi2 camera
# video/x-raw(memory:NVMM): The video format
# width=3264, height=2464: The native video resolution comming from the camera (full resolution) >> grab the video at high resolution
# format=NV12: Camera feed is formatted as NV12
# framerate=21/1: Is 21 frames per second, can run @ 60fps, but not in full resolution
# nvvidconv flip-method='+str(flip)+: Flip the screen
# video/x-raw, width='+str(dispW)+', height='+str(dispH)+': Set the camera Display Window (Width & Height) >> display in low resolution
# format=BGRx: Format the stream as Blue, Green, Red
# appsink: The library needs to be able to pull the frames out of the stream. appsink makes the frames available to openCV
camSet ='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=6/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# Next, we will need to create a camera virtual object named "cam" for the raspi2 cam
cap = cv2.VideoCapture(camSet)


def empty(a):
    pass

# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale", "Result", 400, 1000, empty)
cv2.createTrackbar("Neig", "Result", 20, 50, empty)
cv2.createTrackbar("Min Area", "Result", 3000, 100000, empty)
cv2.createTrackbar("Brightness", "Result", 160, 255, empty)

# LOAD THE CLASSIFIERS DOWNLOADED
cascade = cv2.CascadeClassifier(path)

while True:
    # SET CAMERA BRIGHTNESS FROM TRACKBAR VALUE
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)
    # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # DETECT THE OBJECT USING THE CASCADE
    scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Result") / 1000)
    neig = cv2.getTrackbarPos("Neig", "Result")
    objects = cascade.detectMultiScale(gray, scaleVal, neig)
    # Display the number of detected objects
    cv2.putText(img, str(len(objects)), (305, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(img, "Number of Cats detected: ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (330, 50), (255, 0, 0), 2)
    # DISPLAY THE DETECTED OBJECTS
    for (x, y, w, h) in objects:
        area = w * h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, objectName, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            roi_color = img[y:y + h, x:x + w]

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Clean up the program, let go off the camera, re-run the program could fail otherwise
cv2.destroyAllWindows()  # Close all windows
