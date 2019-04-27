import os
import cv2
import numpy as np
import io
from google.cloud import vision


dataset_path = 'data/faces/'
skip=0
face_data=[]

#cap = cv2.VideoCapture(dataset_path+'video.mp4') 
cap = cv2.VideoCapture(0)

file_name=input("Enter the name of person: ")
i = 0
os.mkdir(dataset_path+'students/'+file_name)

while True:
    i+=1
    ret,frame =cap.read()
    if ret==False:
        continue
    client = vision.ImageAnnotatorClient()
    cv2.imwrite("data/extra/frame.jpg", frame) 
    path="data/extra/frame.jpg"
    with io.open(path,'rb')  as image_file:
        content =image_file.read()
    image = vision.types.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    #likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE','LIKELY', 'VERY_LIKELY')
    offset=10
    for face in faces:
        #print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        #print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        #print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
        #vertices = (['({},{})'.format(vertex.x, vertex.y)
                    #for vertex in face.bounding_poly.vertices]:
                        #print(type(vertex))
        b =[]
        for vertex in face.bounding_poly.vertices:
            b.append(vertex)
        x_i=int(b[0].x)
        x_f=int(b[2].x)
        y_i=int(b[0].y)
        y_f=int(b[2].y)
        
        face_section = frame[y_i:y_f,x_i:x_f]
        face_section = cv2.resize(face_section,(100,100))
     
        face_data.append(face_section)
        print(len(face_data))

    cv2.imshow("Frame_s",face_section)
    cv2.imshow("Frame",frame)

    cv2.imwrite(dataset_path+'students/'+file_name+'.jpg', face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('x'):
        break


print("Data Successfully saved")  

cap.release()
cv2.destroyAllWindows()
