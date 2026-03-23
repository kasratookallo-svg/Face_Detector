import cv2

path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(path)
capture = cv2.VideoCapture(0)


while True:
    ret, frame = capture.read()
    if not ret:
        break


    faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for idx , (x,y,w,h) in enumerate(faces):
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_crop = frame[y:y + h, x:x + w]

        cv2.imshow(f"Face { idx + 1 }", face_crop)

    cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) == 27: # Exit Button
        break
capture.release()
cv2.destroyAllWindows()

