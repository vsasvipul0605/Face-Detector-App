import cv2
from random import randrange
# load some already trained data on face front view from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img = cv2.imread('pari1.jpg')

# convert it to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect face
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
print(face_coordinates)

# draw rectangles around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w,y+h), (randrange(128,256),randrange(128,256),randrange(128,256)), 5)

cv2.imshow('Clever Programmer Face Detector',img)
cv2.waitKey()
print("Hey Hi!!")