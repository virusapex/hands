import cv2
import mediapipe as mp
import os
import numpy as np


def image_processed(file_path):

    hand_img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        print(data)
        data = str(data)
        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []
        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []
        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])

        return clean

    except:
        return np.zeros([1, 63], dtype=int)[0]


mypath = 'dataset'
file_name = open('dataset.csv', 'a')

for each_folder in os.listdir(mypath):
    for each_number in os.listdir(mypath + '/' + each_folder):

        label = each_folder
        file_loc = mypath + '/' + each_folder + '/' + each_number
        data = image_processed(file_loc)

        try:
            for id, i in enumerate(data):
                if id == 0:
                    print(i)

                file_name.write(str(i))
                file_name.write(',')

            file_name.write(label)
            file_name.write('\n')

        except:
            file_name.write('0')
            file_name.write(',')

            file_name.write('None')
            file_name.write('\n')

file_name.close()
print('Датасет готов.')
