import cv2
import time
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

import threading

subscription_key=""
endpoint=""
face_client=FaceClient(endpoint, CognitiveServicesCredentials(subscription_key))

class FaceDetector(threading.Thread):

	def __init__(self, threadID, name):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.org = (50, 30)
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.font_scale = 1
		self.color = (0, 255, 0)
		self.thickness = 2
		self.age = ""
		self.gender = ""
		self.frame2 = None
		self.frame = None
		self.cap = cv2.VideoCapture()
		self.cap.open(0)
		self.counter = 0

	def run(self):
		_, self.frame = self.cap.read()
		self.frame2 = self.frame.copy()
		while (True):
			_, frame = self.cap.read()
			self.frame = frame.copy()
			frame = cv2.putText(frame, "Real Time", self.org, self.font, self.font_scale, self.color, self.thickness, cv2.LINE_AA)
			frame_full = cv2.hconcat([frame, self.frame2])
			cv2.imshow(self.name, frame_full)
			cv2.waitKey(1)
			if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) <1:
				break
		cv2.destroyAllWindows()

	def detect_faces(self, local_image):
		face_attributes = ['emotion', 'age', 'gender']
		detected_faces = face_client.face.detect_with_stream(local_image, return_face_attributes=face_attributes, detection_model='detection_01')
		return detected_faces


	def detector(self):
		emotions_ref = ["neutral", "sadness", "happiness", "disgust", "contempt", "anger", "surprise", "fear"]
		emotions_found  = []
		while (True):
			time.sleep(1)
			frame = self.frame.copy()
			cv2.imwrite('test.jpg', frame)
			local_image = open('test.jpg', "rb")
			faces = self.detect_faces(local_image)
			if (len(faces)>0):
				age = faces[0].face_attributes.age
				gender = faces[0].face_attributes.gender
				gender = (gender.split('.'))[0]
				emotion = self.get_emotion(faces[0].face_attributes.emotion)
				if(emotion[0] in emotions_ref):
					self.counter += 1
					emotions_ref.remove(emotion[0])
				left, top, width, height = self.getRectangle(faces[0])

				frame = cv2.rectangle(frame, (left, top), (left+width, top+height+100), (255, 0, 0), 3)
				frame = cv2.rectangle(frame, (left, top+height), (left+width, top+height+100), (255, 0, 0), cv2.FILLED)
				frame = cv2.putText(frame, "age: "+str(int(age)), (left, top+height+20), self.font, 0.6, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, "gender: "+str(gender), (left, top+height+40), self.font, 0.6, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, "emotion: ", (left, top+height+60), self.font, 0.6, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, str(emotion[0]), (left, top+height+80), self.font, 0.6, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, "Face Detection", self.org, self.font, self.font_scale, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, "#emotions : "+str(self.counter), (400, 30), self.font,  self.font_scale, self.color, self.thickness, cv2.LINE_AA)
				self.frame2 = frame

	def get_emotion(self, emotion_obj):
	    emotion_dict = dict()
	    emotion_dict['anger'] = emotion_obj.anger
	    emotion_dict['contempt'] = emotion_obj.contempt
	    emotion_dict['disgust'] = emotion_obj.disgust
	    emotion_dict['fear'] = emotion_obj.fear
	    emotion_dict['happiness'] = emotion_obj.happiness
	    emotion_dict['neutral'] = emotion_obj.neutral
	    emotion_dict['sadness'] = emotion_obj.sadness
	    emotion_dict['surprise'] = emotion_obj.surprise
	    emotion_name = max(emotion_dict, key=emotion_dict.get)
	    emotion_confidence = emotion_dict[emotion_name]
	    return emotion_name, emotion_confidence


	def getRectangle(self, faceDictionary):
	    rect = faceDictionary.face_rectangle
	    left = rect.left
	    top = rect.top
	    width = rect.width
	    height = rect.height
	    return left, top, width, height


detector = FaceDetector(1, "Face Detection - Azure")
detector.start()
time.sleep(0.5)
detector.detector()
