import cv2,os,urllib.request,time,datetime
import numpy as np
from django.conf import settings

class Classify(object):
    def __init__(self):
        self.confidenceDef       = 0.5
        prototxtFile        = os.path.join(settings.BASE_DIR,"static/MobileNetSSD_deploy.prototxt.txt")
        modelFile           = os.path.join(settings.BASE_DIR,"static/MobileNetSSD_deploy.caffemodel")

        self.classesMobileNetSSD = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        self.colorsMobileNetSSD = np.random.uniform(0, 255, size=(len(self.classesMobileNetSSD), 3))
        self.net = cv2.dnn.readNetFromCaffe(prototxtFile, modelFile)

        self.video_capture = cv2.VideoCapture(0)

    def __del__(self):
        self.video_capture.release()

    def analyzeFrame(self, frame, displayBoundingBox = True, displayClassName = True, displayConfidence = True):
        lables=[]
        (H, W) = frame.shape[:2]
        mobileNetSSDImgSize = (300, 300)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, mobileNetSSDImgSize), 0.007843, mobileNetSSDImgSize, 127.5)

        self.net.setInput(blob)
        detections = self.net.forward()        

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidenceDef:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                if self.classesMobileNetSSD[idx] not in lables:
                    lables.append(self.classesMobileNetSSD[idx])
                if(displayBoundingBox):
                    cv2.rectangle(frame, (startX, startY), (endX, endY), self.colorsMobileNetSSD[idx], 2)
                if(displayClassName and displayConfidence):                    
                    label = "{}: {:.2f}%".format(self.classesMobileNetSSD[idx], confidence * 100)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorsMobileNetSSD[idx], 2)
                elif (displayClassName):                    
                    label = str(f"{self.classesMobileNetSSD[idx]}")
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorsMobileNetSSD[idx], 2)
        return frame, lables

    def get_frame_clas(self):
        camera_Width  = 640 # 1024 # 1280 # 640
        camera_Heigth = 480 # 780  # 960  # 480
        frameSize = (camera_Width, camera_Heigth)

        (W, H) = (None, None)
        detectionEnabled = True
        
        ret, frameOrig = self.one_frame()
        frame = cv2.resize(frameOrig, frameSize)

        if(detectionEnabled):
            frame, lables=self.analyzeFrame(frame)

        if 'person' in lables:
            self.save_image(frame, 'class')
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        
    def get_frame_move(self):
        frame1, frame2 = self.two_frame()

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
        if len(contours)>=1:
            self.save_image(frame2, 'move')
        ret, jpeg = cv2.imencode('.jpg', frame1)
        return jpeg.tobytes()

    def get_frame(self):
        ret, frame = self.one_frame()
        self.save_image(frame, 'norm')
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def one_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            return ret,frame
        else:
            self.one_frame()
    
    def two_frame(self):
        ret1, frame1 = self.video_capture.read()
        ret2, frame2 = self.video_capture.read()
        if ret1 == True and ret2== True:
            return frame1, frame2
        elif ret1 == True and ret2 != True:
            ret2, frame2 = self.one_frame()
            return frame1, frame2
        elif ret1 != True and ret2 == True:
            ret2, frame1 = self.one_frame()
            return frame1, frame2
        else:
            self.two_frame()


    def save_image(self, frame, mode):
        if mode == 'norm':
            storage_path='images/normal/'
        elif mode == 'class':
            storage_path='images/classified/'
        elif mode == 'move':
            storage_path='images/movement/'
        now = datetime.datetime.now()
        hr=str(now.time().hour)
        mi=str(now.time().minute)
        sec=str(now.time().second)
        mis=str(now.time().microsecond)
        path=hr+mi+sec+mis
        filed=storage_path+path+'.jpeg'
        a=cv2.imwrite(filed,frame)