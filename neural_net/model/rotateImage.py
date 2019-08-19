import shutil

import cv2, os

import random
from random import randint
def mover():
    gest_folder = "D:/Descargas/NBS2/train"
    directorio_llegada= "D:/Descargas/NBS2/valid"
    images_labels = []
    images = []
    labels = []
    for g_id in os.listdir(gest_folder):
        i=0
        list=[]
        for imagen in os.listdir(gest_folder+"/"+g_id):
            list.append(imagen)



        print(len(list))
        x=0
        aleat=[]

        while(x<=200):
            indice=randint(0,len(list)-1)
            if(indice not in aleat):
                aleat.append(indice)
                path = gest_folder + "/" + g_id + "/" + list[indice]
                new_path = directorio_llegada + "/" + g_id + "/" + list[indice]
                print(path + "\n")
                print(new_path)
                shutil.move(path, new_path)
                x = x + 1














mover()
