from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from controler.camera import Classify
# Create your views here.

def index(request):
    return render(request, 'home.html')

def gen(work):
    obj=Classify()
    if work==0:
        while True:
            frame = obj.get_frame()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    elif work==1:
        while True:
            frame = obj.get_frame_move()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    elif work==2:
        while True:
            frame = obj.get_frame_clas()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(0),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def movement(request):
    return StreamingHttpResponse(gen(1),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def classify(request):
    return StreamingHttpResponse(gen(2),
                    content_type='multipart/x-mixed-replace; boundary=frame')