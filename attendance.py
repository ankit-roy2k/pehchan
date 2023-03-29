import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
from playsound import playsound
import datetime
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json

path = 'static\\uploads'
images = []
classNames = []
myList = os.listdir(path)
#countList = []
regList = set()

class STOP_FLAG:
    stopFlag = True
    
    @staticmethod
    def stopAttendance():
        STOP_FLAG.stopFlag = False
    
    @staticmethod
    def startAttendance():
        STOP_FLAG.stopFlag = True
		

# List out the sample faces with names
def sampleFaces(path):
    for img in myList:
        curImg = cv2.imread(f'{path}/{img}')
        images.append(curImg)
        classNames.append(os.path.splitext(img)[0])
    return images,classNames

# Encoding the sample images
def findEncodingds(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Mark Attendance
def markAttendance(name,countList):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        countFace = 0
        for row in myDataList:
            entry = row.split(',')
            regList.add(entry[0])
            countFace = countList.count(name)
            if (name[0] not in regList) & (countFace >= 15):
                now = datetime.datetime.now()
                time = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name[0]},{name[1]},{name[3]},{name[2]},{time},{time}')
                print('done')
                playsound('thanks.mp3')
                for c in countList:
                    # print(countList)
                    if c == name:
                        countList.remove(c)
                # print(countList,"final countList")
            else:
                if (entry[0] == name[0]) & (countFace >= 15):
                    now = datetime.datetime.now()
                    timeLater = now.strftime('%H:%M:%S')
                    nowTime = timeLater.split(':')
                    entryTime = entry[-1].split(':')
                    if (int(nowTime[0]) > int(entryTime[0])):
                        f.writelines(f',{timeLater}')
                        playsound('thanks.mp3')
                        for c in countList:
                            if c == name:
                                countList.remove(c)
                    elif (int(nowTime[1]) - int(entryTime[1])) >= 5:    #Register the attendance at least after 5 mins of the last record
                        f.writelines(f',{timeLater}')
                        playsound('thanks.mp3')
                        for c in countList:
                            if c == name:
                                countList.remove(c)

    return countList

# Recognition module
def recognise(encodeListKnown, classNames, video_capture, isLive, countList):
    regList = set()
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    try:
            ret_recognize, frame_recognize = video_capture.read()
            small_frame = cv2.resize(frame_recognize, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
                    name = ""
                    face_distances = face_recognition.face_distance(encodeListKnown , face_encoding)
                    best_match_index = np.argmin(face_distances)
                    # if np.int16(best_match_index) <= 6:
                    if matches[best_match_index]:
                        name = classNames[best_match_index]

                    print(name)
                    face_names.append(name)

            process_this_frame = not process_this_frame

            print(face_names)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
    
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if isLive:
                    cv2.rectangle(frame_recognize, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame_recognize, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    student_detail = name.split("_")
                    first_name = student_detail[1]
                    #print(firstName + str(datetime.datetime.today()))
                    cv2.putText(frame_recognize, first_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    countList.append(student_detail)
                    markAttendance(student_detail, countList)
                else:
                    cv2.rectangle(frame_recognize, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame_recognize, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    student_detail = name.split("_")
                    first_name = student_detail[1]
                    #print(firstName + str(datetime.datetime.today()))
                    cv2.putText(frame_recognize, "fake", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.imshow("Video", frame_recognize)
                    
    except Exception as e:
        print(e)


def stop():
    STOP_FLAG.stopAttendance()

def checkTime():
    check = datetime.datetime.now()
    timeNow = check.strftime('%H:%M:%S')
    checkNow = timeNow.split(':')
    return int(checkNow[0])

def saveFile():
    today = datetime.date.today()
    savePath = 'Everyday_Attendance/'+str(today)+'_Attendance.csv'
    savePathJson = 'Everyday_Attendance/'+str(today)+'_Attendance.json'

    with open('Attendance.csv','r+') as f3:
        data3 = f3.readlines()
        with open('convert.csv','r+') as f4:
            for c in data3:
                content = c.split(',')
                # print(content)
                if content != ['\n'] :
                    f4.writelines(f'\n{content[0]},{content[1]},{content[2]},{content[3]},{content[4]},{content[-1]}')
    df = pd.read_csv('convert.csv')
    savePath = 'Everyday_Attendance/'+str(today)+'_Attendance.csv'
    savePathJson = 'Everyday_Attendance/'+str(today)+'_Attendance.json'
    df.to_csv(savePath)
    df.to_json(savePathJson)

    print('-------------- Attendance Saved For Today --------------')

    # Reset the convert.csv file for next day use.
    filename = "convert.csv"
    f = open(filename, "w+")
    f.close()

    # Reset the Attendance.csv file for next day use
    with open("Attendance.csv", "r") as f2:
        data2 = f2.readlines()
    with open("Attendance.csv", "w") as f2:
        for row2 in data2:
            if row2.strip("\n") == "Registration No. , Name , Branch , Semester , Time , Last Seen":
                f2.write(row2)

def live(encodeListKnown, classNames):
    countList = []
    root_dir = os.getcwd()
    # Load Face Detection Model
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    # Load Anti-Spoofing Model graph
    json_file = open('antispoofing_models/antispoofing_model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load antispoofing model weights 
    model.load_weights('antispoofing_models/antispoofing_model.h5')
    print("Model loaded from disk")

    video_capture= cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        try:
            ret,frame = video_capture.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:  
                face = frame[y-5:y+h+5,x-5:x+w+5]
                resized_face = cv2.resize(face,(160,160))
                resized_face = resized_face.astype("float") / 255.0
                # resized_face = img_to_array(resized_face)
                resized_face = np.expand_dims(resized_face, axis=0)
                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(resized_face)[0]
                if preds > 0.5:
                    recognise(encodeListKnown, classNames, video_capture, False, countList)
                else:
                    recognise(encodeListKnown, classNames, video_capture, True, countList)
            key = cv2.waitKey(1)
            if cv2.waitKey(1)== ord('q') or STOP_FLAG.stopFlag != True:
                cv2.destroyAllWindows()
                break
        except Exception as e:
            pass
    video_capture.release()        
    cv2.destroyAllWindows()


# Main Function
def main():
    t = checkTime()
    if t < 24:
        STOP_FLAG.startAttendance()
        images, classNames = sampleFaces(path)
        encodeListKnown = findEncodingds(images)
        print('--------------------Encoding Complete------------------')
        live(encodeListKnown, classNames)
    else:
        stop()
        saveFile()
