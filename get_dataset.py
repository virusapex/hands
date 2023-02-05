import cv2
from pathlib import Path


class_id = input ('Введи название жеста: ')
Path('dataset/' + class_id).mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Не могу открыть поток...")
    exit()

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удаётся получить кадры. Выходим из программы...")
        break

    i += 1
    if i % 5 == 0:
        cv2.imwrite('dataset/' + class_id + '/' + str(i) + '.jpg', frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q') or i > 500:
        break

cap.release()
cv2.destroyAllWindows()
