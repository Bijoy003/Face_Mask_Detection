import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import cv2

font = cv2.FONT_HERSHEY_COMPLEX

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')
incorrect_mask = np.load('incorrect_mask.npy')
print(with_mask.shape)
print(without_mask.shape)
print(incorrect_mask.shape)
with_mask = with_mask.reshape(796, 100 * 100 * 3)
without_mask = without_mask.reshape(1016, 100 * 100 * 3)
incorrect_mask = incorrect_mask.reshape(486, 100 * 100 * 3)
print(with_mask.shape)
print(without_mask.shape)
x = np.r_[with_mask, without_mask, incorrect_mask]
print(x.shape)
labels = np.zeros(x.shape[0])
labels[796:] = 1.0
labels[1812:] = 2.0

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=.20, )
print(x_train.shape)
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
print(x_train[0])
print(x_train.shape)

# support vector machine sklearn
# support vector classification sklearn
svm = SVC()
svm.fit(x_train, y_train)
# pca
x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

print(accuracy_score(y_test, y_pred))

category = {0: 'Mask', 1: 'No Mask', 2: 'Incorrect Mask'}
video = cv2.VideoCapture(0)
haar = cv2.CascadeClassifier("data.xml")
while True:
    flag, frame = video.read()
    if flag:
        faces = haar.detectMultiScale(frame)
        for x, y, w, h in faces:
            face = frame[y:y + h, x:x + w:]
            face = cv2.resize(face, (100, 100))
            face = face.reshape(1, -1)
            face = pca.transform(face)
            pred = svm.predict(face)
            n = category[int(pred)]
            print(n)
            if int(pred) == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.putText(frame, n, (x, y), font, 1, (0, 0, 255), 2)
            elif int(pred) == 2:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
                cv2.putText(frame, n, (x, y), font, 1, (0, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(frame, n, (x, y), font, 1, (0, 255, 0), 2)

        # print(faces)
        cv2.imshow("Mask Detector", frame)
        k = cv2.waitKey(2)
        if k == 27:
            break
video.release()
cv2.destroyAllWindows()
