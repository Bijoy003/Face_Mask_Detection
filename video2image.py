# video to face image
import cv2

video = cv2.VideoCapture('./videos/v8.MOV')

haar = cv2.CascadeClassifier("data.xml")
cnt = 5000
while True:
    ret, frame = video.read()
    faces = haar.detectMultiScale(frame)
    for x, y, w, h in faces:
        cnt = cnt + 1
        name = './images/with_mask/' + str(cnt) + '.jpg'
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        face = frame[y:y + h, x:x + w:]
        face = cv2.resize(face, (100, 100))
        cv2.imwrite(name, face)

    # print(faces)
    cv2.imshow("Mask Detector", frame)
    k = cv2.waitKey(2)
    if k == 27:
        break
video.release()
cv2.destroyAllWindows()
