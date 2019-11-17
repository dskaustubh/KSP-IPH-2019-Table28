import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import pickle
import matplotlib as plt

def get_encoded_faces():
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


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file(img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are
    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    img = cv2.imread(im, 1)
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img,face_locations)
    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        # matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        # use the known face with the smallest distance to the new face
        # face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        ans= face_recognition.compare_faces(faces_encoded,face_encoding,tolerance=.5)
        k=[i for i, x in enumerate(ans) if x]
        if  k:
            name=known_face_names[k[0]]
        face_names.append(name)
        print(name,im)
        if name=='Unknown':
            pass
        
        else:
            ka=cv2.imread(im)
            wa=cv2.imread("ArrestPerson_images/"+name)
            k=cv2.resize(ka,(270,360))
            w=cv2.resize(wa,(270,360))
            cv2.imshow('Missing',k)
            cv2.imshow('Arrested',w)
            cv2.waitKey()

        
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
        classify_face("faces1/"+f)
    except:
        pass

