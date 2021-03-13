from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from controler.camera import VideoCamera, Movement, Classify
from PIL import Image
import binascii
# Create your views here.

def index(request):
	return render(request, 'home.html')

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')

def movement(request):
	return StreamingHttpResponse(gen(Movement()),
					content_type='multipart/x-mixed-replace; boundary=frame')

def classify(request):
	return StreamingHttpResponse(gen(Classify()),
					content_type='multipart/x-mixed-replace; boundary=frame')