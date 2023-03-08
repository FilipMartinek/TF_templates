import cv2, os, sys, pickle, threading
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import models, process_data

#USE GUIDE:
#python3.8 determine_face/train.py [optional: MODEL_NUM] [optional: "ow" for overwrite data] [optional: DATASET] [optional: MAXLEN] [optional: GPU_NUM] [optional: "m" for multiple gpu use]
#DATASET can be UTKFace or IMDB_WIKI
#COLORTYPE can be bgr, rgb or bw
#GPU_NUM will be taken as number of gpus with "m" paramater


#select gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #first gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" #second gpu


#create a filedir vat and get models
filedir = os.getcwd() + "/determine_face/"
models = models.get_models()


#create vars for number of epochs and batch size
EPOCHS = 300
BATCH_SIZE = 32


#train, save model and history
def train(Model, data_train, data_test, labels_train, labels_test, filename):

    #create checkpointer and early stop, then add them to the callback list
    Checkpoint = ModelCheckpoint(filename, monitor="val_loss", verbose=1,save_best_only=True, save_weights_only=False, mode="auto", save_freq="epoch")
    Early_stop = EarlyStopping(patience=150, monitor="val_loss",restore_best_weights=True),
    callbacks = [Checkpoint, Early_stop]


    #train model and save training history
    start = datetime.now()
    History = Model.fit(imgs_train, labels_train, batch_size=BATCH_SIZE, validation_data=(imgs_test, labels_test),epochs=EPOCHS, callbacks=[callbacks])
    time = datetime.now() - start
    history = History.history
    history.update({"time" : time})
    pickle.dump(history, open(f"{filename[:3]}_history.pickle", "wb"), pickle.HIGHEST_PROTOCOL)

    #save complete model
    Model.save(filename)


#if program is ran
if __name__ == "__main__":
    
    #default parameters
    DATASET = "UTKFace"
    OW = False
    MAXLEN = sys.maxsize
    MULTIPLEGPUS = False
    GPUNUM = "-1"


    #try to get a model num paramater
    try:
        #get model num
        temp = 1
        MODEL_NUM = int(sys.argv[temp])
        temp += 1


        #try to get other paramaters
        try:
            if sys.argv[temp] == "ow":
                OW = True
                temp += 1
            if sys.argv[temp] == "UTKFace" or sys.argv[temp] == "IMDB_WIKI":
                DATASET = sys.argv[temp]
                temp += 1
            try:
                MAXLEN = int(sys.argv[temp])
                temp += 1
            except ValueError:
                pass
        except IndexError:
            pass
        try:
            GPUNUM = sys.argv[temp]
        except IndexError:
            pass

        #select gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUNUM

        #get model data
        Model, RES, COLORS, COLOR_TYPE = models[MODEL_NUM]
        if DATASET == "IMDB_WIKI":
            imgs_train, imgs_test, labels_train, labels_test = process_data.process_data_IMDB_WIKI(RES, COLOR_TYPE, OW, MAXLEN)
        elif DATASET == "UTKFace":
            imgs_train, imgs_test, labels_train, labels_test = process_data.process_data_UTKFace(RES, COLOR_TYPE, OW, MAXLEN)

        #train, save, and evaluate model
        train(Model, imgs_train, imgs_test, labels_train, labels_test)
        Model.save(f"determine_face/models/model{MODEL_NUM}.h5")
        Model.evaluate(imgs_test,labels_test)

    except Exception:

        #try to get other paramaters
        try:
            if sys.argv[temp] == "ow":
                OW = True
                temp += 1
            if sys.argv[temp] == "UTKFace" or sys.argv[temp] == "IMDB_WIKI":
                DATASET = sys.argv[temp]
                temp += 1
            try:
                MAXLEN = int(sys.argv[temp])
                temp += 1
            except ValueError:
                pass
        except IndexError:
            pass
        try:
            GPUNUM = sys.argv[temp]
            temp += 1
        except IndexError:
            pass
        try:
            if sys.argv[temp] == "m":
                MULTIPLE_GPUS = True
        except IndexError:
            pass

        #define thread for multiple gpus
        def thread(models, gpu):

            #select gpu
            #os.environ["CUDA_VISIBLE_DEVICES"] = gpu


            #go through all the models and train them
            for i, model_data in enumerate(models):
                
                #get model data
                print(model_data)
                Model, RES, COLORS, COLOR_TYPE = model_data
                if DATASET == "IMDB_WIKI":
                    imgs_train, imgs_test, labels_train, labels_test = process_data.process_data_IMDB_WIKI(RES, COLOR_TYPE, OW, MAXLEN)
                elif DATASET == "UTKFace":
                    imgs_train, imgs_test, labels_train, labels_test = process_data.process_data_UTKFace(RES, COLOR_TYPE, OW, MAXLEN)

                #train, save, and evaluate model
                train(Model, imgs_train, imgs_test, labels_train, labels_test, f"model{i}")
                Model.save(f"determine_face/models/model{i}.h5")
                Model.evaluate(imgs_test,labels_test)
        

        #if there's only one gpu
        if not(MULTIPLEGPUS):
            thread(models, GPUNUM)
        else:
            chunk = int(len(models)) // GPUNUM
            for chunknum, i in enumerate(range(0, len(models), chunk)):
                #create and start a thread
                x = threading.Thread(target=thread, args=[models[i:i+chunk], str(chunknum)])
                x.start()
            #last thread if models is not divisible by GPUNUM
            if int(len(models)) % GPUNUM > 0:
                thread(models[i:])
