import cv2
import numpy as np
import os 
def distance(v1, v2):
    # Eucledian 
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    
    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
cap = cv2.VideoCapture(0)
dataset_path = './data/'

face_data =[]
labels=[]

class_id = 0
names = {}
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #Create a mapping btw class_id and name
        names[class_id] = fx[:-4]
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #Create Labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    cv2.imwrite("/home/chirag/attendance/frame.jpg", frame) 
    #from PIL import Image, ImageDraw
    path="/home/chirag/attendance/frame.jpg"
    with io.open(path,'rb')  as image_file:
        content =image_file.read()
    image = vision.types.Image(content=content)
    #image.source.image_uri = uri
    #print(type(image))
    response = client.face_detection(image=image)
    faces = response.face_annotations
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE','LIKELY', 'VERY_LIKELY')
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
        out = knn(trainset,face_section.flatten())
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x_i,y_i),(x_f,y_f),(0,255,255),2)

    cv2.imshow("Faces",frame)

    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()