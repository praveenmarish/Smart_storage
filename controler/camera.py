import cv2,os,urllib.request,time
import numpy as np
from django.conf import settings

# MobileNetSSD Settings
confidenceDef       = 0.5
thresholdDef        = 0.3
prototxtFile        = os.path.join(settings.BASE_DIR,"static/MobileNetSSD_deploy.prototxt.txt")
modelFile           = os.path.join(settings.BASE_DIR,"static/MobileNetSSD_deploy.caffemodel")

global classesMobileNetSSD, colorsMobileNetSSD, net
classesMobileNetSSD = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
colorsMobileNetSSD = np.random.uniform(0, 255, size=(len(classesMobileNetSSD), 3))
net = cv2.dnn.readNetFromCaffe(prototxtFile, modelFile)

class VideoCamera(object):
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def __del__(self):
        self.video_capture.release()

    def analyzeFrame(self, frame, displayBoundingBox = True, displayClassName = True, displayConfidence = True):
        (H, W) = frame.shape[:2]
        mobileNetSSDImgSize = (300, 300)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, mobileNetSSDImgSize), 0.007843, mobileNetSSDImgSize, 127.5)

        net.setInput(blob)
        detections = net.forward()        

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidenceDef:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                if(displayBoundingBox):
                    cv2.rectangle(frame, (startX, startY), (endX, endY), colorsMobileNetSSD[idx], 2)
                if(displayClassName and displayConfidence):                    
                    label = "{}: {:.2f}%".format(classesMobileNetSSD[idx], confidence * 100)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorsMobileNetSSD[idx], 2)
                elif (displayClassName):                    
                    label = str(f"{classesMobileNetSSD[idx]}")
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorsMobileNetSSD[idx], 2)
        return frame


    def get_frame(self):
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        # Camera Settings
        camera_Width  = 640 # 1024 # 1280 # 640
        camera_Heigth = 480 # 780  # 960  # 480
        frameSize = (camera_Width, camera_Heigth)

        (W, H) = (None, None)
        detectionEnabled = True
        
        ret, frameOrig = self.video_capture.read()
        frame = cv2.resize(frameOrig, frameSize)

        if(detectionEnabled):
            frame=self.analyzeFrame(frame)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


class Movement(object):
    def __init__(self):
        self.video_capture1 = cv2.VideoCapture(0)

    def __del__(self):
        self.video_capture1.release()

    def get_frame(self):
        ret, frame1 = self.video_capture1.read()
        ret, frame2 = self.video_capture1.read()

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            '''if cv2.contourArea(contour) < 900:
                continue'''
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame1)
        return jpeg.tobytes()
