import cv2
import os
import numpy as np
from PIL import Image
from keras import models
from pygame import mixer
import time

#Load the saved model
model = models.load_model('models\\phone_model.h5')
classes=['no_phone','talking_phone','texting_phone']
cam = cv2.VideoCapture(0)
img_size1 = 128
img_size2 = 128


mixer.init()
sound = mixer.Sound('alarm.wav')
path = os.getcwd()
score=0


while True:  #kameradan sınırsız değer icin sınırsız döngü
    ret,frame = cam.read() #whether the camera is working or not, returns a numpy array in the frame variable                                                 kameranın çalışıp çalışmadığı, kare değişkeninede bir numpy array döndürüyor
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# convert gray image  
    new_img = cv2.resize(gray_img,(img_size2,img_size1))
    test_data = np.array(new_img).reshape(-1,img_size2,img_size1,1); 

          
    print(test_data.shape)
    preds = model.predict(test_data)
    predict= np.argmax(preds)

 #predict score and write score
    if(predict== 1 or predict==2):
        score=score+1
        cv2.putText(frame,classes[predict],(50,460),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)

    else:
        score=score-1
        cv2.putText(frame,classes[predict],(50,460),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
      
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(50,440), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    
    if(score>60):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except: 
            pass
    cv2.imshow("Ekran",frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

