# -*- coding: utf-8 -*-
"""HCR_FINAL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tNjTAe6tSFHQsooSUbxBrmJi9sfy8GrZ
"""

"""Mapping of class labels with charactes
0->48='0'
1->49='1'
2->50='2'
3->51='3
4->52='4
5->53='5
6->54='6
7->55='7
8->56='8
9->57='9

10->65='A'
11->66='B'
12->67='C'
13->68='D'
14->69='E'
15->70='F'
16->71='G'
17->72='H'
18->73='I'
19->74='J'
20->75='K'
21->76='L'
22->77='M'
23->78='N'
24->79='O'
25->80='P'
26->81='Q'
27->82='R'
28->83='S'
29->84='T'
30->85='U'
31->86='V'
32->87='W'
33->88='X'
34->89='Y'
35->90='Z'

36->97='a'
37->98='b'
38->99='c'
39->100='d'
40->101='e'
41->102='f'
42->103='g'
43->104='h'
44->105='i'
45->106='j'
46->107='k'
47->108='l'
48->109='m'
49->110='n'
50->111='o'
51->112='p'
52->113='q'
53->114='r'
54->115='s'
55->116='t'
56->117='u'
57->118='v'
58->119='w'
59->120='x'
60->121='y'
61->122='z'

"""
import cv2
import numpy as np
import tensorflow as tf

char={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'c',39:'d',40:'e',41:'f',42:'g',43:'h',44:'i',45:'j',46:'k',47:'l',48:'m',49:'n',50:'o',51:'p',52:'q',53:'r',54:'s',55:'t',56:'u',57:'v',58:'w',59:'x',60:'y',61:'z'}

number_of_classes = 62

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.2))

model.add(Flatten())

model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(.2))

model.add(Dense(units = 62, activation = 'softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights('D:/vs/OASIS/Downloads/Handwritten-Character-Recognition-main/Handwritten-Character-Recognition-main/weights.h5')


from keras.preprocessing import image
def predict(img_path):
    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(28,28))
#     plt.imshow(img)
    test_img_arr = image.img_to_array(img)
    test_img_arr = np.expand_dims(test_img_arr, axis = 0)
    test_img_arr=test_img_arr/255.0
    prediction = model.predict(test_img_arr)
    pred_char=char[np.argmax(prediction[0])]
    return pred_char