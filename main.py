import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
from PIL import Image, ImageDraw

# from PIL import ImageGrab

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)


  # get fileName from user 
filepatho = input("Enter Classcode: ") 
filepath= filepatho+".csv"
now1=date.today()


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


"""def markAttendance(name):
    with open('Attendances.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        dtList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            dtList.append(entry[1])
            now=date.today()
            if (name not in nameList)and(now not in dtList) :
                #now = datetime.now()
                #now=date.today()
                # dtString = now.strftime('%H:%M:%S')
                dtString=now
                f.writelines(f'\n{name},{dtString}')"""
"""def markAttendance(name):
    with open('Attendances.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  line:
            nowd=date.today()
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string},{nowd}')
            # f.writelines(f'\n{name},{dt_string},{nowd}')"""

def markAttendance(name):
    already_in_file = set()
    #with open('Attendances.csv', "r") as g:       # just read
    with open(filepath, "r") as g:
        for line in g:
            already_in_file.add(line.split(",")[0])
            #already_in_file.add(line.split())

# process your current entry:
    if name  and now1 not in already_in_file:
        #with open('Attendances.csv', "a") as g:   # append
        with open(filepath, "a") as g:
            #now1=date.today()
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            g.writelines(f'\n{name},{dtString},{now1}')                

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')


#for webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# static image
"""file_name = "wasii.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(encodeListKnown, face_encoding)

    name = "Unknown"

    face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = classNames[best_match_index]

    # Draw a box around the face using the Pillow module
    cv2.rectangle(unknown_image_to_draw,(left, top), (right, bottom), (0,255,0),3 )
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 255))
    cv2.putText(unknown_image_to_draw,name,(left,top-20), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2,cv2.LINE_AA)
    print(name)
    markAttendance(name)"""

# display(pil_image)
#cv2.imshow(unknown_image_to_draw,file_name)