import os, random, time, logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import scipy.io as sio
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam
from keras.models import Model
from keras.backend import clear_session
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Input, Concatenate, Convolution1D, Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from threading import Thread
import mfcc 

'''
## Training mode 線程 ##

'''
class DNNDeepLearningThread(Thread):
    # dl_process = pyqtSignal(int) ## 輸出training 進度 ##
    # dl_accuracy = pyqtSignal(float) ## 輸出準確率 ##
    # dl_cm = pyqtSignal(np.ndarray) ## 輸出Confusion matrix ##
    # dl_message = pyqtSignal(str) ## 輸出字串顯示狀態 ##

    def __init__(self, *args, **kwargs):
        super(DNNDeepLearningThread, self).__init__(*args, **kwargs)
       
    def run(self, dl_pro, classes, get_directory_path):
        global layers_num
        # global dl_pro, classes, get_directory_path, cm, Accuracy
        # layers_num, globalpath,
        while dl_pro:
            dl_pro = False
            clear_session()
            layers_num = 2
            train_data1 = None
            train_labels = None
            try:
                for d in classes:
                    sourcePath = os.path.join(get_directory_path, "feature")
                    # print(sourcePath)
                    fileNumber = [k for k in os.listdir(sourcePath)]
                    for k in fileNumber:
                        cur_data1 = sio.loadmat(os.path.join(sourcePath, k)).get("data_Acous")
                        cur_labels = sio.loadmat(os.path.join(sourcePath, k)).get("label")
                        cur_labels = np.transpose(cur_labels, (1,0))       
                        if train_labels is None or train_data1 is None:
                            train_labels = cur_labels
                            train_data1 = cur_data1
                        else:
                            train_labels = np.vstack((train_labels, cur_labels))
                            train_data1 = np.vstack((train_data1, cur_data1))

                train_labels = to_categorical(train_labels)

                self.MdPath = sourcePath + r'/../Result/result'                
                
                rn = random.sample(range(train_data1.shape[0]),train_data1.shape[0])
                
                train_data1=train_data1[rn]
                train_labels=train_labels[rn]
                
                indata1=train_data1[0:round(len(rn)*0.9)]
                inlabel=train_labels[0:round(len(rn)*0.9)]
        
                outdata1=train_data1[round(len(rn)*0.9):len(rn)]
                outlabel=train_labels[round(len(rn)*0.9):len(rn)]
        
                inputs1 = Input(shape=[indata1.shape[1]])
                x1 = Dense(100)(inputs1)
                x1 = Dropout(0.4)(x1)
                
                for layer in range(1,layers_num):
                    x1 = Dense(100)(x1)
                    x1 = Dropout(0.3)(x1)
                

                outlayer = Dense(inlabel.shape[1], activation='softmax')(x1)
                self.model = Model(inputs=[inputs1], outputs=outlayer)
                                
                
                self.model.summary()
                nadam=Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004);
                OptimIz=nadam;
                self.model.compile(
                						loss='categorical_crossentropy', 
                                        metrics=['acc'],
                						optimizer=OptimIz)
                checkpointer = ModelCheckpoint(
                						filepath=self.MdPath+".hdf5",
                						monitor="acc",
                						mode="max",
                						verbose=2,
                						save_best_only=True)
                
                # self.dl_process.emit(0)
                
                epochs = 50

                for i in range(0,epochs):
                    self.model.fit([indata1], inlabel, batch_size=64, epochs=1, verbose=2,callbacks=[checkpointer],validation_split=0.3,shuffle=True)
                    # dl_bar = (i+1)/epochs*100
                    # print(dl_bar)
               
                Prediction = self.model.predict([outdata1], batch_size=64, verbose=1)
                Results = np.argmax(Prediction, axis=1)
    
                Accuracy = accuracy_score(np.argmax(outlabel, axis=1), Results)
                
                # print(Accuracy)
                # self.dl_accuracy.emit(Accuracy)
     
                cm = confusion_matrix(np.argmax(outlabel, axis=1), Results)

                with open(self.MdPath+".json", "w") as f:
                    f.write(self.model.to_json())
                    print('\nThe model is successfully saved.\n')
                    return cm, Accuracy
                     # self.dl_message.emit("The model is successfully saved.")               

                
            except NameError:
                logging.error('\x1b[31mThe file seems to have something wrong. Please check the data.\x1b[0m')
                # self.dl_message.emit("The file seems to have something wrong. Please check the data.")
            # except TypeError:
            #     self.dl_message.emit("The file seems to have something wrong. Please check the data.")
    
class CNNDeepLearningThread:
    # dl_process = pyqtSignal(int) ## 輸出training 進度 ##
    # dl_accuracy = pyqtSignal(float) ## 輸出準確率 ##
    # dl_cm = pyqtSignal(np.ndarray) ## 輸出Confusion matrix ##
    # dl_message = pyqtSignal(str) ## 輸出字串顯示狀態 ##

    def __init__(self,*args, **kwargs):
        super(CNNDeepLearningThread, self).__init__(*args, **kwargs)

    
    def input_prepare(self, inload, sequence_length):  
        sequence_length = sequence_length-1
        data = list()
        for i in range(inload.shape[0]):
            if i-sequence_length < 0:
                for j in range(-(i-sequence_length)):
                    if j ==0:
                        pre = inload[0,:]
                    else:
                        pre = np.vstack((pre, inload[0,:]))
                data.append(np.vstack((pre, inload[0:i+1,:]))	)
            else:
                data.append(inload[i-sequence_length:i+1,:])
        	
        indata = np.array(data).astype('int16')
        return indata

    def run(self, dl_pro, classes, get_directory_path):
        global layers_num
        # global dl_pro, classes, layers_num, globalpath, get_directory_path
        while dl_pro:
            dl_pro = False
            clear_session()
            layers_num = 2
            train_data1 = None
            train_labels = None
            try:          
                for d in classes:
                    sourcePath = os.path.join(get_directory_path, "feature")
                    print(sourcePath)
                    fileNumber = [k for k in os.listdir(sourcePath)]
                    
                    for k in fileNumber:
                        cur_data1 = sio.loadmat(os.path.join(sourcePath, k)).get("data_Acous")
                        cur_labels = sio.loadmat(os.path.join(sourcePath, k)).get("label")
                        cur_labels = np.transpose(cur_labels, (1,0))       
                        if train_labels is None or train_data1 is None:
                            train_labels = cur_labels
                            train_data1 = cur_data1
                        else:
                            train_labels = np.vstack((train_labels, cur_labels))
                            train_data1 = np.vstack((train_data1, cur_data1))

                train_labels = to_categorical(train_labels)
                
                self.MdPath = sourcePath + r'/../Result/result'
                sequence_length = 10                
                
                               
                rn = random.sample(range(train_data1.shape[0]),train_data1.shape[0])
                train_data1 = self.input_prepare(train_data1,sequence_length)
                
                train_data1=train_data1[rn]
                train_labels=train_labels[rn]
                    
                indata1=train_data1[0:round(len(rn)*0.9)]
                inlabel=train_labels[0:round(len(rn)*0.9)]
        
                outdata1=train_data1[round(len(rn)*0.9):len(rn)]
                outlabel=train_labels[round(len(rn)*0.9):len(rn)]
            
                
                inputs1 = Input(shape=[indata1.shape[1], indata1.shape[2]])
                x1 = Convolution1D(64, 3, activation='relu')(inputs1)
                x1 = Dropout(0.4)(x1)
                   
                for layer in range(1,layers_num):
                    x1 = Convolution1D(64, 3, activation='relu')(x1)
                    x1 = Dropout(0.3)(x1)
                    
                x1 = Flatten()(x1)
                outlayer = Dense(inlabel.shape[1], activation='softmax')(x1)
                self.model = Model(inputs=[inputs1], outputs=outlayer)
                    
                    
                self.model.summary()
                nadam=Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004);
                OptimIz=nadam;
                self.model.compile(
                    						loss='categorical_crossentropy', 
                                            metrics=['acc'],
                    						optimizer=OptimIz)
                checkpointer = ModelCheckpoint(
                    						filepath=self.MdPath+".hdf5",
                    						monitor="acc",
                    						mode="max",
                    						verbose=2,
                    						save_best_only=True)
                    
                # self.dl_process.emit(0)
                    
                
                epochs = 50
                    
                for i in range(0,epochs):
                    self.model.fit([indata1], inlabel, batch_size=64, epochs=1, verbose=2,callbacks=[checkpointer],validation_split=0.3,shuffle=True)
                    # self.dl_process.emit((i+1)/epochs*100)
        
                Prediction = self.model.predict([outdata1], batch_size=64, verbose=1)
                Results = np.argmax(Prediction, axis=1)
                Accuracy = accuracy_score(np.argmax(outlabel, axis=1), Results)
                # print(Accuracy)
                # print(Results)
                # threshold = 0.6
                # total_predict = len(Results)
                # if Results.count(0)/total_predict<=threshold:
                #     print(0)
                # else:
                #     print(1)
                # self.dl_accuracy.emit(Accuracy)
                cm = confusion_matrix(np.argmax(outlabel, axis=1), Results)
                
                
                with open(self.MdPath+".json", "w") as f:
                    f.write(self.model.to_json())
                    print('\nThe model is successfully saved.\n')
                    return cm, Accuracy
                    # self.dl_message.emit("The model is successfully saved.")
            
            except NameError:
                logging.error('\x1b[31mThe file seems to have something wrong. Please check the data.\x1b[0m')
            #     self.dl_message.emit("The file seems to have something wrong. Please check the data.")
            # except TypeError:
            #     self.dl_message.emit("The file seems to have something wrong. Please check the data.")