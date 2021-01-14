import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

#sınıflandırıcı ayarları
face = cv2.CascadeClassifier('haarcascade\\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade\\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade\\haarcascade_righteye_2splits.xml')

label=['Closed','Open']

model = load_model('models\\drowsy_model.h5')

path = os.getcwd()
cap = cv2.VideoCapture(0) #kamera açma
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #gri kullanarak yüz algılama
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

  
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    
    #yüz için sınır kutuları çizimi
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        
    #gözün görüntü verilerini çıkarma
    for (x,y,w,h) in right_eye: 
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY) #renkli görüntüyü gri tonlama
        r_eye = cv2.resize(r_eye,(24,24)) #24*24 piksel boyutlandırma
        r_eye= r_eye/255  #yakınsama için normalizasyon işlemi
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            label='Open' 
        if(rpred[0]==0):
            label='Closed'
        break

    for (x,y,w,h) in left_eye: 
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255 
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            label='Open'   
        if(lpred[0]==0):
            label='Closed'
        break

 #skor tahmini ve sonucu ekrana yazdırma
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>20):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:   isplaying = False
        
   
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()