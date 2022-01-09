import csv
import numpy as np
import tensorflow
from keras.layers import Dense, Input
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

inputs = []
teacher = []
with open("data.csv", newline='') as csv_file:
    spam_reader = csv.reader(csv_file, delimiter=',')
    next(spam_reader)
    for row in spam_reader:
        tab_teacher = 0
        tab_input = list(map(int, row[0:7]))
        tab_teacher = int(row[7])
        inputs.append(tab_input)
        teacher.append(tab_teacher)

inputs_np = np.array(inputs)
teacher_np = np.array(teacher)

inputs_np = np.reshape(inputs_np, (len(inputs_np), 1, 7))
teacher_np = to_categorical(teacher_np, dtype='int32')
teacher_np = np.reshape(teacher_np, (len(teacher_np), 1, 10))

# print("inputs_np: '{}'\n".format(inputs_np))
# print("teacher_np: '{}'\n".format(teacher_np))


inputs = []
teacher = []
with open("test.csv", newline='') as csv_file:
    spam_reader = csv.reader(csv_file, delimiter=',')
    next(spam_reader)
    for row in spam_reader:
        tab_input = np.zeros((1, 7))
        tab_teacher = 0
        tab_input = list(map(int, row[0:7]))
        tab_teacher = row[7]
        inputs.append(tab_input)
        teacher.append(tab_teacher)

input_test = np.array(inputs)
teacher_test = np.array(teacher)

input_test = np.reshape(input_test, (len(input_test), 1, 7))
teacher_test = to_categorical(teacher_test, dtype='int32')
teacher_test = np.reshape(teacher_test, (len(teacher_test), 1, 10))

#print("input_test: '{}'\n".format(input_test))
#print("teacher_test: '{}'\n".format(teacher_test))

sgd = SGD(lr=0.05)

model = Sequential()
model.add(Dense(8, input_dim=7, activation='relu', input_shape=(1, 7)))

model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_history = model.fit(inputs_np, teacher_np, epochs=5000,
                          validation_data=(input_test, teacher_test))

#   test du fonctionnement de notre réseau de neurones

test = np.array([[[0, 1, 1, 0, 0, 1, 0]], [[0, 1, 0, 0, 0, 0, 0]]])
predictions = model.predict(test)
print(f'Predictions : {predictions}')
for v in predictions:
    result = np.where(v[0] == np.amax(v[0]))
    print('Indices :', result[0])
    print('Tab retourné :', result)
    print(f'Valeur : {np.amax(v[0])}')


model_history.history['accuracy']

#************** COURBES **************
f = plt.figure(1)
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()

# summarize history for loss
g = plt.figure(2)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.show()
