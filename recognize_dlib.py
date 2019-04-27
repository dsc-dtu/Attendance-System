import face_recognition
import cv2
import numpy as np
from google.cloud import vision
import io

names = ['alwin']
s_image = []
s_encoding = []

for i in range(len(names)):
    s_image.append(face_recognition.load_image_file("data/faces/students/"+names[i]+".png"))
    s_encoding.append(face_recognition.face_encodings(s_image[-1])[0])


known_face_encodings = s_encoding
known_face_names = names


process_this_frame = True


#````````````````````````````````````````````````````````````````````````````````````
video_capture = cv2.VideoCapture("video.mp4") 


while True:
    ret,frame = video_capture.read()
    if ret == False:
        continue

    client = vision.ImageAnnotatorClient()

    if process_this_frame:

        new_faces = []
        face_encodings = []
        face_locations = []
        cv2.imwrite("data/extra/frame.jpg", frame) 
        path="data/extra/frame.jpg"
        with io.open(path,'rb')  as image_file:
            content =image_file.read()
        image = vision.types.Image(content=content)

        response = client.face_detection(image=image)
        faces = response.face_annotations
        for face in faces:
            b =[]

            for vertex in face.bounding_poly.vertices:
                b.append(vertex)
            x_i=int(b[0].x)
            x_f=int(b[2].x)
            y_i=int(b[0].y)
            y_f=int(b[2].y)
        
            face_section = frame[y_i:y_f,x_i:x_f]
            face_section = cv2.resize(face_section,(100,100))
            face_locations.append((y_f,x_f,y_i,x_i))
            face_encodings.append(face_recognition.face_encodings(face_section)[0])


        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
    

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    

    cv2.imshow('Video', frame)

    process_this_frame = not process_this_frame

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

video_capture.release()
cv2.destroyAllWindows()
