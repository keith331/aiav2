# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:11:20 2023

@author: Bcsic_abcd
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical

Prediction = model.predict([test_data1], batch_size=64, verbose=1)



       #          self.DNNDeepLearningThread = DNNDeepLearningThread()
       # L984     cm, accuracy = self.DNNDeepLearningThread.run(dl_pro, self.classes, self.get_task_directory_path)
       
       
       
        # conceal_module.py L.106
if 1:
    if 1:
        if 1:
            if 1:        
                 Prediction = model.predict([outdata1], batch_size=64, verbose=1)
                 Results = np.argmax(Prediction, axis=1)
    
                 Accuracy = accuracy_score(np.argmax(outlabel, axis=1), Results)

     
                 cm = confusion_matrix(np.argmax(outlabel, axis=1), Results)
                 
                 ###
                 
# T_lps.shape
# Out[73]: (876, 129)
                 
                 T_lps_input = T_lps[np.newaxis,:,:,np.newaxis]
                 Prediction = model.predict(T_lps_input, batch_size=64, verbose=1)



                rn = random.sample(range(train_data1.shape[0]),train_data1.shape[0])
                
                train_data1=train_data1[rn]
                train_labels=train_labels[rn]
                
                indata1=train_data1[0:round(len(rn)*0.9)]
                inlabel=train_labels[0:round(len(rn)*0.9)]
        
                outdata1=train_data1[round(len(rn)*0.9):len(rn)]
                outlabel=train_labels[round(len(rn)*0.9):len(rn)]        
                
                Prediction = self.model.predict([outdata1], batch_size=64, verbose=1)
                Results = np.argmax(Prediction, axis=1)
    
                Accuracy = accuracy_score(np.argmax(outlabel, axis=1), Results)
                
                # print(Accuracy)
                # self.dl_accuracy.emit(Accuracy)
     
                cm = confusion_matrix(np.argmax(outlabel, axis=1), Results)
                

Accuracy = accuracy_score(np.argmax(BB, axis=1), label) 
cm = confusion_matrix(np.argmax(BB, axis=1), label)       
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

####

filepath = r'D:\AI-A job\aiav1.0 ckt4 multu type show keith2_\aiav1.0 ckt4 multu type show_py\use_data\aia5_15\feature'
matfile = filepath + r'\0_1.mat'

cur_data1 = sio.loadmat(matfile).get("data_Acous")


                rn = random.sample(range(cur_data1.shape[0]),cur_data1.shape[0])
                
                train_data1=train_data1[rn]
                train_labels=train_labels[rn]
                
                indata1=train_data1[0:round(len(rn)*0.9)] # 0:104
                inlabel=train_labels[0:round(len(rn)*0.9)]
        
                outdata1=train_data1[round(len(rn)*0.9):len(rn)] # 104:115
                outlabel=train_labels[round(len(rn)*0.9):len(rn)]








Predict = model.predict(cur_data1[104:115], batch_size=64, verbose=1)
