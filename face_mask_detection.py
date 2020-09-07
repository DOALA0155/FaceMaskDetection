import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

capture = cv2.VideoCapture(0)

face_log = []

while True:
  ret, image = capture.read()
  image = cv2.flip(image, 1)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  eyes = eye_cascade.detectMultiScale(gray)

  if len(eyes) == 0:
    face_log.append(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Face is not detected", (10, 30), font, 1, (0, 0, 0), 2)
  else:
    if len(faces) == 0:
      face_log.append(1)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(image, "Thank you for your participation on wearing masks.", (10, 30), font, 1, (0, 0, 0), 2)

    else:
      face_log.append(2)
      if face_log[-10:].count(2) >= 9:
        cv2.putText(image, "Alert", (10, 80), font, 3, (0, 0, 255), 2)

      else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "You must wear a mask.", (10, 30), font, 1, (0, 0, 255), 2)
        for (x, y, w, h) in faces:
          cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = image[y:y+h, x:x+w]

  cv2.imshow('Mask Detection', image)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break


capture.release()
cv2.destroyAllWindows()
