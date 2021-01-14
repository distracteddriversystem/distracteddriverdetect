#importing libraries for the data processing and model.
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import load_model

directory ='data//train'
test_directory ='data//test'
#random_test = 'random_test'
classes = ['closed','open']
# defining a shape to be used for our models.
img_size1 = 24
img_size2 = 24
# creating a training dataset.
training_data = []
i = 0
def create_training_data():
    for category in classes:
        path = os.path.join(directory,category)
        class_num = classes.index(category)      
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img_array,(img_size2,img_size1))
            training_data.append([
                new_img,class_num])
create_training_data()
print(len(training_data))


# Creating a test dataset.
testing_data = []
i = 0
def create_testing_data():        
    for img in os.listdir(test_directory):
        img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array,(img_size2,img_size1))
        testing_data.append([img,
            new_img])
create_testing_data()
print(len(testing_data))


random.shuffle(training_data)
x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)
x[0].shape
X = np.array(x).reshape(-1,img_size2,img_size1,1)
X.shape,X[0].shape
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=50)
Y_train = np_utils.to_categorical(y_train,num_classes=2)
Y_test = np_utils.to_categorical(y_test,num_classes=2)


model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',input_shape=(24,24,1)))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units = 512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units = 256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5)]
batch_size = 64
n_epochs = 10
results = model.fit(x_train,Y_train,batch_size=batch_size,epochs=n_epochs,verbose=1,validation_data=(x_test,Y_test),callbacks=callbacks)

model.save("models//drowsy_model.h5")

# Plot training & validation accuracy values
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

loaded_model = load_model('models//drowsy_model.h5')

#confusion-matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
fig = plt.figure(figsize=(10, 10))
y_pred = loaded_model.predict(x_test) 
Y_pred = np.argmax(y_pred, 1) 
Yy_test = np.argmax(Y_test, 1) 
mat = confusion_matrix(Yy_test, Y_pred)
# Plot Confusion matrix
sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
plt.show();

#counting accuracy-sensivity-specificity value
print("")
print("")
print("")
total1=sum(sum(mat))
accuracy=(mat[0,0]+mat[1,1])/total1
print ('Accuracy : ', accuracy)
print('---------------------------------')
precision = mat[0,0]/(mat[0,0]+mat[1,0])
print('Precision : ', precision )
print('---------------------------------')
recall=mat[0,0]/(mat[0,0]+mat[0,1])
print('Recall : ', recall )










