# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:59:27 2018

@author: Dalia
"""
import numpy as np
import glob
import os
import cv2
# from HOG import HOG
import pickle
from project import get_features
from sklearn.svm import SVC


#todo
#dictionary for each class to get closed_loop, connected_comp

# ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ى لا
dict= {1:[0,1], 2:[0,1], 3:[0,2],4:[0,3], 5:[0,1], 6:[0,0],7:[0,1], 8:[0,0], 9:[0,1],10:[0,0],
 11:[0,1], 12:[0,0],13:[0,3], 14:[1,0], 15:[1,1],16:[1,0], 17:[1,1], 18:[0,0],19:[0,1], 20:[1,1],
 21:[1,2],22:[0,1], 23:[0,0], 24:[1,0],25:[0,1], 26:[1,0], 27:[1,0],28:[0,0], 29:[1,0]}


TrainingSamples = []
# TestingSamples = []

labels = []
model = SVC(kernel='rbf',C=1.0,gamma='auto')

# svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                 svm_type = cv2.SVM_C_SVC,
#                 C=2.67, gamma=5.383 )


def trainSVM(Dir,Type,closed_loop,connected_comp):
    
    print("\n\nStart Training")

    i=0
    for filename in glob.glob(os.path.join(Dir, '*.png')):
        # img = cv2.imread(filename, 1)
        feature = get_features(filename,closed_loop,connected_comp)
        TrainingSamples.append(feature)
        labels.append(Type)
        i+=1
        if(i%1000==0):
            if Type !=-1:
                print(str(i)+" positive images are trained")
            elif Type ==-1:
                print(str(i)+" negative images are trained")    

###################################################################

def takeDataset():
    Dir=""
    Dir=input("Enter training data directory: ")
    while (Dir.lower()!=str("STOP").lower() ):
        Type=int(input("Enter the type of training data (from 1 to 29): "))
        # closed_loop=int(input("Enter the closed_loop feature (1:exist, 0:not exist): "))
        # connected_comp=int(input("Enter the number of connected_components feature: "))
        closed_loop = dict[Type][0]
        connected_comp = dict[Type][1]
        trainSVM(Dir,Type,closed_loop,connected_comp)
        print("\n\nTo break the loop and save the model ENTER STOP")
        Dir=input("Enter training data directory: ")

###################################################################

def training():
    model.fit(TrainingSamples,labels)
    # Set up SVM for OpenCV 3
    # svm = cv2.ml.SVM_create()
    # svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setC(0.1)
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))

    # svm = cv2.SVM()
    # svm.train(TrainingSamples, cv2.ml.ROW_SAMPLE,labels)
    # svm.save('svm_data.yml')

    print("SUCCESS!!! Training is completed !!!!")

###################################################################

def saveModel():
    SVMModelName=input("Enter trained model name in format 'filename.pkl': ")
    file = open(SVMModelName, "wb")
    file.write(pickle.dumps(model))
    file.close()

###################################################################

def testing():
    correct = 0
    Dir=""
    Dir=input("Enter testing data directory: ")
    model_pkl = pickle.load(open('alphabet.pkl', 'rb'))
    total=0
    while (Dir.lower()!=str("STOP").lower() ):
        Type=int(input("Enter the type of testing data (from 1 to 29): "))
        # closed_loop=int(input("Enter the closed_loop feature (1:exist, 0:not exist): "))
        # connected_comp=int(input("Enter the number of connected_components feature: "))
        TestingSamples = []
        i=0
        for filename in glob.glob(os.path.join(Dir, '*.png')):
            closed_loop = dict[Type][0]
            connected_comp = dict[Type][1]
            feature = get_features(filename,closed_loop,connected_comp)
            TestingSamples.append(feature)
            i+=1
            if(i%1000==0):
                print(str(i)+" images are tested")

        correct+= predict(Type,model_pkl,TestingSamples)
        total+= len(TestingSamples)
        print("\n\nTo break the loop and see accuracy ENTER STOP")
        Dir=input("Enter testing data directory: ")
    
    print(round(correct*100.0/total,2))
        

###################################################################

def predict(Type,model,TestingSamples):
    # alif_baa = 'alif_baa.pkl'
    # alif = pickle.load(open(pkl, 'rb'))
    mask = []
    for T in TestingSamples:
        result = model.predict([T])
        if(result == Type):
            mask.append(1)
        else:
            mask.append(0)

    correct = np.count_nonzero(mask)
    return correct
    # return (correct/len(mask))

    
###################################################################33

# takeDataset()
# training()
# saveModel()
testing()
