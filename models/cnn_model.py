"""
Created on Mon Sep 12 13:57:09 2022

@author: sbplab
"""
import os
import os.path as osp
import sys
import numpy as np
import itertools 
import time
from PySide2.QtCore import Qt, Signal, Slot, QObject

import tensorflow as tf
# from tensorflow.keras import backend

from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Model ,model_from_json

from tensorflow.keras.layers import Multiply, Add, Embedding

from tensorflow.keras.layers import Input, Activation, Dense

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, GlobalAveragePooling1D

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,TerminateOnNaN

from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam, Nadam, Adamax

from models import utils
# os.environ["CUDA_VISIBLE_DEVICES"]="0" #0: GPU2, 1: GPU3, 2: GPU0, 3: GPU1

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Gpu_Control = tf.compat.v1.GPUOptions(allow_growth=True) # G_RAM will auto control it's range
# sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options=Gpu_Control))

class CNNModel(QObject):

    sig_finished = Signal()
    sig_cm = Signal(np.ndarray)

    def __init__(self, model_path, cm_xlabel):
        super(CNNModel, self).__init__()

        self.model_path = model_path
        self.source0 = osp.join(self.model_path, 'training', 'zero')
        self.source1 = osp.join(self.model_path, 'training', 'one')
        self.test0 = osp.join(self.model_path, 'training', 'zero_testing')
        self.test1 = osp.join(self.model_path, 'training', 'one_testing')
        self.result_path = osp.join(self.model_path, 'result')
        self.T_label2 = cm_xlabel

    def run(self):

        sr = 48000
        frame_size = 768
        overlap = 384
        fft_size = 768
        # result_path = r'./Model/Rnn5ep_forcpu'
        result_path = str(self.result_path) + '\\' + 'Rnn5ep'
        epochs = 2
        batch_size = 12
        seq_num = 385
        print('you are here')
        # source0 = utils.auto_list(r'C:\Users\chihk\projects\apuiv2\use_data\test_model\training\zero',nat_sort=True)
        # source1 = utils.auto_list(r'C:\Users\chihk\projects\apuiv2\use_data\test_model\training\one',nat_sort=True)
        source0 = utils.auto_list(self.source0)
        source1 = utils.auto_list(self.source1)

        X_wavs0 = utils.load_wavs(source0, sr)
        X_wavs1 = utils.load_wavs(source1, sr)

        X_lps0, X_info0 = utils.lps_extract(X_wavs0, frame_size, overlap, fft_size, to_list = False)
        X_lps1, X_info1 = utils.lps_extract(X_wavs1, frame_size, overlap, fft_size, to_list = False)
        X_lps0, X_lps1= X_lps0.T, X_lps1.T

        X_lps0 = utils.time_splite(X_lps0,time_len=seq_num, padding = False)
        X_lps0 = X_lps0[:,:,:,np.newaxis]
        X_lps1 = utils.time_splite(X_lps1,time_len=seq_num, padding = False)
        X_lps1 = X_lps1[:,:,:,np.newaxis]

        X_label = np.hstack([np.ones(len(X_lps0))*0, np.ones(len(X_lps1))*1]);
        X_label = to_categorical(X_label)

        X_lps = np.concatenate((X_lps0, X_lps1), axis=0)

        Inputs = Input(shape=(None, seq_num, 1), name="input")

        x = Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation='relu', padding = 'valid')(Inputs)
        x = MaxPooling2D(pool_size = (2, 2), padding = 'valid')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation='relu', padding = 'valid')(x)
        x = MaxPooling2D(pool_size = (2, 2), padding = 'valid')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation='relu', padding = 'valid')(x)
        x = MaxPooling2D(pool_size = (2, 2), padding = 'valid')(x)
        x = Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation='relu', padding = 'valid')(x)
        x = GlobalAveragePooling2D()(x)
        Outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=Inputs, outputs=Outputs)
        model.summary()


        model.compile(loss='categorical_crossentropy',
                      optimizer=Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
                      metrics=['acc'])

        checkpointer = ModelCheckpoint(
                                filepath=result_path+".hdf5",
                                monitor="acc",
                                mode="max",
                                verbose=1,
                                save_best_only=True)
        StartTime= time.time()

        history = model.fit(X_lps, X_label,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[checkpointer],
                            validation_split=0.001)
        EndTime= time.time()
        with open(result_path+".json", "w") as f:
            f.write(model.to_json())
            

        # test0 = utils.auto_list(r'C:\Users\chihk\projects\apuiv2\use_data\test_model\training\zero_testing',True);
        test0 = utils.auto_list(self.test0)
        T0_wavs = utils.load_wavs(test0, sr)
        T0_lps_all, T0_info = utils.lps_extract(T0_wavs, frame_size, overlap, fft_size, to_list = True)

        # test1 = utils.auto_list(r'C:\Users\chihk\projects\apuiv2\use_data\test_model\training\one_testing',True);
        test1 = utils.auto_list(self.test1)
        T1_wavs = utils.load_wavs(test1, sr)
        T1_lps_all, T1_info = utils.lps_extract(T1_wavs, frame_size, overlap, fft_size, to_list = True)

        T_lps_all =  T0_lps_all+T1_lps_all
        T_label = np.hstack([np.ones(len(T0_lps_all))*0, np.ones(len(T1_lps_all))*1]);
        T_label = to_categorical(T_label)

        # T_label2 = np.vstack([np.ones(len(T0_lps_all))*0, np.ones(len(T1_lps_all))*1]);
        T_label2 = self.T_label2

        Rlts = None
        for eval_id, T_lps in enumerate(T_lps_all):
            T_lps = T_lps.T[np.newaxis,:,:,np.newaxis]
            results = model.evaluate(T_lps, T_label[eval_id:eval_id+1], batch_size=128, verbose=0)
            prediction = model.predict(T_lps, verbose=0)
            print("Class:{:01d} Accuracy:{:02.2f}%".format(np.argmax(prediction),results[-1]*100))
            if Rlts is None:
                Rlts = prediction
            else:
                Rlts = np.vstack((Rlts,prediction))
        print(100*Rlts)

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        #Accuracy = accuracy_score(T_label2,np.argmax(Rlts, axis=1)) 
        cm = confusion_matrix(T_label2,np.argmax(Rlts, axis=1))    
        
        print(np.argmax(Rlts, axis=1))   
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)       

        self.sig_cm.emit(cm)      
        self.sig_finished.emit()


if __name__ == '__main__': 

    cnn = CNNModel(r'C:\Users\chihk\projects\apuiv2\use_data\test_model')
    cnn.run()