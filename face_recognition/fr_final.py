import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import pickle


def get_encoded_faces_wanted():
    """
    looks through the faces folder and encodes all
    the faces
    :return: dict of (name, image encoded)
    """
    encoded = {}
    # c=0
    with open("wanted.pkl","rb") as f:
        encoded=pickle.load(f)
    return encoded

def get_encoded_faces_arrested():
    """
    looks through the faces folder and encodes all
    the faces
    :return: dict of (name, image encoded)
    """
    encoded = {}
    # c=0
    with open("arrested.pkl","rb") as f:
        encoded=pickle.load(f)
    return encoded

def get_encoded_faces_deaths():
    """
    looks through the faces folder and encodes all
    the faces
    :return: dict of (name, image encoded)
    """
    encoded = {}
    # c=0
    with open("unwanted_death.pkl","rb") as f:
        encoded=pickle.load(f)
    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file(img)
    encoding = fr.face_encodings(face)[0]

    return encoding

def find_in_db(im,known_face_names,unknown_face_encodings,face_names,faces_encoded,str):
        for face_encoding in unknown_face_encodings:
            name = "Unknown"
            ans= face_recognition.compare_faces(faces_encoded,face_encoding,tolerance=.5)
            k=[i for i, x in enumerate(ans) if x]
            if  k:
                name=known_face_names[k[0]]
            face_names.append(name)
            print(name,im,"\t",str)
            if name=='Unknown':
                pass
            else:
                ka=cv2.imread(im)
                wa=cv2.imread(str+"/"+name)
                k=cv2.resize(ka,(270,360))
                w=cv2.resize(wa,(270,360))
                cv2.imshow('Missing',k)
                cv2.imshow(str,w)
                cv2.waitKey()

def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are
    :param im: str of file path
    :return: list of face names
    """
    faces_death = get_encoded_faces_deaths()
    faces_arrested = get_encoded_faces_arrested()
    faces_wanted = get_encoded_faces_wanted()

    faces_encoded_death = list(faces_death.values())
    known_face_names_death = list(faces_death.keys())

    faces_encoded_arrested = list(faces_arrested.values())
    known_face_names_arrested = list(faces_arrested.keys())

    faces_encoded_wanted = list(faces_wanted.values())
    known_face_names_wanted = list(faces_wanted.keys())

    img = cv2.imread(im, 1)
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img,face_locations)
    face_names = []
    find_in_db(im,known_face_names_death,unknown_face_encodings,face_names,faces_encoded_death,"unnatural_death_images/unnatural_death_images")
    find_in_db(im,known_face_names_arrested,unknown_face_encodings,face_names,faces_encoded_arrested,"ArrestPerson_images")
    find_in_db(im,known_face_names_wanted,unknown_face_encodings,face_names,faces_encoded_wanted,"wanted")













     
        
fk = []
# for (dirpath, dirnames, filenames) in os.walk("./missing_pics/"):
#     fk.extend(filenames)
#     break
fk=os.listdir("./faces1")
# classify_face("test3.jpg")
# for f in fk:
#     classify_face(f)
# print(fk)
for f in fk:
    try:
        # print("eden")
        classify_face("faces1/"+f)
    except:
        pass
# classify_face("test3.jpg")
