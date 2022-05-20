import cv2
import os   #create a directory ,check,delete all
dataset="dataset"
name="Elon"

#creating directory
path=os.path.join(dataset,name) #join dataset/sankaran directory
if not os.path.isdir(path):
    os.mkdir(path)

#to save the image in specific width and height
(width,height)=(130,100)

#assiging the .xml file into the variable
alg="haarcascade_frontalface_default.xml"

#loading the file using cv into haar_cascade variable
haar_cascade=cv2.CascadeClassifier(alg)

#initializing the camera
cam=cv2.VideoCapture(0)
count=1


while count<50:
    print(count)
    _,img=cam.read()
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(grayImg,1.3,4)#to get the co_ordinates from detect multi scale method
    for (x,y,w,h) in face: #obtaining x y w,h axis from face variable
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        faceOnly=grayImg[y:y+h,x:x+w] #to crop only the face
        resizeImg=cv2.resize(faceOnly,(width,height))#resize the image with help of cv
        cv2.imwrite("%s/%s.jpg"%(path,count),resizeImg)
        count+=1
    cv2.imshow("facedetection",img)
    key=cv2.waitKey(10)
    if key==27:
        break
print("image captured successfully")
cam.release()
cv2.destroyAllWindows()
