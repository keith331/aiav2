# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:52:13 2023

@author: sbplab
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:38:49 2022

@author: sbplab
"""

import os
import os.path as osp
import sys
import time
import cv2
import copy
# import keras
from tensorflow import keras
import numpy as np
from numpy import asarray 
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from PySide2.QtCore import Signal, Slot, QObject

from PIL import Image
from tensorflow.keras import backend as K
# from keras.preprocessing import image
from tensorflow.keras.models import Model ,model_from_json
# from keras.utils.np_utils import to_categorical

# sys.path.append("./Function")
from models import utils
os.environ["CUDA_VISIBLE_DEVICES"]="0" #0: GPU2, 1: GPU3, 2: GPU0, 3: GPU1

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
Gpu_Control = tf.compat.v1.GPUOptions(allow_growth=True) # G_RAM will auto control it's range
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options=Gpu_Control))    


class HeatMap(QObject):

    sig_finished = Signal()
    sig_heatmap = Signal(np.ndarray)
    sig_result = Signal(list)

    def __init__(self, model_path, test_file, save_path, sn, is_backtest):
        super(HeatMap, self).__init__()
        self.model_path = model_path
        self.result_path = osp.join(self.model_path, 'result')
        self.test_file = test_file
        self.save_path = save_path
        if is_backtest:
            self.sn = osp.basename(test_file).split('.')[0]
        else:
            self.sn = sn

    def run(self): 
        
        sr = 48000
        frame_size = 768
        overlap = 384
        fft_size = 768
        # model_path = r'./Model/Rnn5ep_forcpu'
        model_path = str(self.result_path) + '\\' + 'Rnn5ep'
        epochs = 2
        batch_size = 12
        seq_num = 385

        selectnum = 0
        result_path = self.save_path
        atten_thresh = 0.85
        
        # test = utils.auto_list(r'C:\Users\chihk\projects\apuiv2\use_data\test_model\training\testwav',True)
        # test = utils.auto_list(self.test_file)
        # print(test[selectnum])

        T_wavs = utils.load_wav(self.test_file, sr)
        print(T_wavs)
        T_lps_all, T_info_all = utils.lps_extract(T_wavs, frame_size, overlap, fft_size, to_list = True)
        
        T_lps = T_lps_all[selectnum].T
        T_info = T_info_all[selectnum]      
        
        T_lps_input = T_lps[np.newaxis,:,:,np.newaxis]

        # for new version grad_cam    
        with open(model_path+'.json', "r") as f:
            model = model_from_json(f.read())
            
        model.load_weights(model_path+'.hdf5')
        # A = model.predict(T_lps_input, batch_size=129, verbose=2, steps=None)
        layer_name2,_,suggest_method = layer_detector(model,-1,'Conv2D')
        grad_cam = Grad_CAM(model, T_lps_input, -1, layer_name2)

        # for old version grad_cam
        # tf.compat.v1.disable_eager_execution()
        # with open(model_path+'.json', "r") as f:
        #     model = model_from_json(f.read())
            
        # model.load_weights(model_path+'.hdf5')
        result = []
        A = model.predict(T_lps_input, batch_size=129, verbose=2, steps=None)
        print(A)
        B = np.argmax(A, axis=1)
        if B==0:
            print('PASS')
            result.append('Pass')
        if B==1:
            print('Fail')
            result.append('Fail')

        score = A[0][B][0] * 100
        score = round(score,1)
        score = str(score) + '%'
        result.append(score)
        print(score)

        # layer_name3,_,suggest_method = layer_detector(model,-1,'Conv2D')
        # grad_cam = Grad_CAM_old(model, T_lps_input, -1, layer_name3)
        
        atten_area,atten_bound = atten_extract(grad_cam,atten_thresh)
        extract_lps,extract_cam = crop_by_atten(T_lps,grad_cam,atten_area)
        
        Img_result=get_atten_img(T_lps,grad_cam,extract_lps,extract_cam,atten_bound)
        Wav_result={'origin_lps':T_lps.T,'atten_lps':extract_lps,
                    'origin_info':T_info,'sampling_rate':sr, 
                    'item_type':'waveform'}
        
        # save_result(Img_result,test[selectnum],result_path)
        # save_result(Wav_result,test[selectnum],result_path)

        # overlay_lps = Img_result['img_lps']
        # overlay_lps = asarray(overlay_lps)

        overlay_lps = save_result(Img_result,self.sn,result_path)
        save_result(Wav_result,self.sn,result_path)

        self.sig_heatmap.emit(overlay_lps)
        self.sig_result.emit(result)
        self.sig_finished.emit()

def layer_detector(model,direction=-1,Target_layer=None):
    
    if direction>0:
        direction,start_index=[1,0]
    else:
        direction,start_index=[-1,-1]
        
    layer_len=direction*(len(model.layers)+1)
    layer_list={'Target_layer':[Target_layer],
                'GAP':['GlobalAveragePooling2D','AveragePooling2D'],
                'Conv':['Conv2D']}

#==============================================================================
    if Target_layer:
        try:
            for layer_index in range(start_index,layer_len,direction):

                layer_type=model.layers[layer_index].__class__.__name__
                
                if any([layer_type in layer_list['Target_layer']]):
                    print('%s detected!'%(layer_type))
                    target_layer_name=model.layers[layer_index].name
                    target_layer_index=layer_index
                    suggest_method=None
                    break
        except:
            print('''Couldn't find %s layer
                  in current model...''' %(Target_layer))
            sys.exit(1)
    else:
        try:
            for layer_index in range(start_index,layer_len,direction):
        #        layer_info=model.layers[layer_index].get_config()
                layer_type=model.layers[layer_index].__class__.__name__
                
                if any([layer_type in layer_list['GAP']]):
                    print('%s detected!'%(layer_type))
                    target_layer_name=model.layers[layer_index + direction].name
                    target_layer_index=layer_index + direction
                    suggest_method='CAM'
                    break
                
                elif any([layer_type in layer_list['Conv']]):
                    print('%s layer detected!' %(layer_type))
                    target_layer_name=model.layers[layer_index].name
                    target_layer_index=layer_index + direction
                    suggest_method='grad_CAM'
                    break
        except:
            print('''Couldn't find Convolution layer 
                  or Global Average Pooling in this model,...''')
            sys.exit(1)
#==============================================================================
    return target_layer_name,target_layer_index, suggest_method

def CAM(input_model, img, clc, layer_name):

    samples,width, height, _ = img.shape
    
    if samples ==1:
        y_c = input_model.output
    
        #Get the input weights to the softmax.
        class_weights = input_model.layers[-1].get_weights()[0]
#        class_weights = input_model.layers[-2].get_weights()[0]
        
        conv_output = input_model.get_layer(layer_name).output
        
        get_output = K.function([input_model.input], [conv_output, y_c])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]
        
        print('Model prediction:')
        if clc == -1:
            clc = np.argmax(predictions)
        print('Predict Class: %d' %(clc))
        
        resize_weights = cv2.resize(class_weights, (class_weights.shape[-1],conv_outputs.shape[-1]), cv2.INTER_LINEAR)
        cam =np.dot(conv_outputs, resize_weights[:, clc])
    #    cam =np.dot(conv_outputs, class_weights[:, clc])
        cam = cam / cam.max()
        cam = cv2.resize(cam, (height, width), cv2.INTER_LINEAR)
        plt.figure()
        plt.title(' Class Activation Map ')
        plt.axis('off')
        plt.imshow(np.flipud(cam.T), cmap='jet', alpha=1)
        return cam
    
def grad_normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

def Grad_CAM(input_model, img, clc, layer_name):

    samples,width, height, _ = img.shape
    
    if samples ==1:

        y_c = input_model.output
        conv_output = input_model.get_layer(layer_name).output
        gradient_function = Model([input_model.input], [conv_output, y_c])
        # [conv_outputs, predictions] = gradient_function([img])
        
        with tf.GradientTape() as gtape:
            # gtape.watch([img])
            [conv_outputs, predictions] = gradient_function(img)
            # predictions = input_model.predict(img)
            if clc == -1:
                clc = np.argmax(predictions)
            loss = predictions[:,clc]

        grads= gtape.gradient(loss,conv_outputs)
        grads_val = grad_normalize(grads)    
        output, grads_val = conv_outputs[0, :], grads_val[0, :, :, :]
        
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)
        
        # Process CAM
        cam = cv2.resize(cam, (height, width), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        # cam = cam / cam.max()
        # plt.figure()
        # plt.title(' Class Activation Map ')
        # plt.axis('off')
        # plt.imshow(np.flipud(cam.T), cmap='jet', alpha=1)
        
        return cam
    
def Grad_CAM_old(input_model, img, clc, layer_name):

    samples,width, height, _ = img.shape
    
    if samples ==1:
        # y_c = input_model.output
        predictions = input_model.predict(img)
        print('Model prediction:')
                    
        if clc == -1:
            clc = np.argmax(predictions)
        print('Predict Class: %d' %(clc))
    
        y_c = input_model.output[0, clc]
        conv_output = input_model.get_layer(layer_name).output
        grads = K.gradients(y_c, conv_output)[0]
        # Normalize if necessary
        grads = grad_normalize(grads)
        gradient_function = K.function([input_model.input], [conv_output, grads])
    
        output, grads_val = gradient_function([img])
        output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        
        # Process CAM
        cam = cv2.resize(cam, (height, width), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        # cam = cam / cam.max()
        # plt.figure()
        # plt.title(' Class Activation Map ')
        # plt.axis('off')
        # plt.imshow(np.flipud(cam.T), cmap='jet', alpha=1)
        
        return cam    

def mat_normalize(mat_input): 

    mat_max,mat_min= np.max(mat_input),np.min(mat_input)
    #code for plotting log power spectrogram
    if mat_min == float("-inf"):
        mat_min=np.nanmin(mat_input[mat_input != float("-inf")])
        mat_input[mat_input==float("-inf")]=mat_min
    #code for plotting log power spectrogram
    normalize_mat=(mat_input-mat_min)/(mat_max-mat_min)

    return normalize_mat

def mat_crop(input_mat,mask_area,fill_val):
    
    atten_mat = copy.deepcopy(input_mat)
    atten_mat[mask_area] = fill_val
                   
    return atten_mat

def atten_extract(cam_mat,threshold):
    
    origin_cam = copy.deepcopy(np.flipud(cam_mat.T))
    atten_area=mat_normalize(origin_cam)>=threshold
       
    y_diff = np.diff(atten_area,n=1,axis=0)
    y_diff = np.insert(y_diff,-1,y_diff[-1,:],axis=0)
    x_diff = np.diff(atten_area,n=1,axis=1)
    x_diff = np.insert(x_diff,-1,x_diff[:,-1],axis=1)

    atten_bound = np.logical_or(x_diff,y_diff)
               
    return atten_area, atten_bound

def trans_to_img(input_mat,cmap_name,is_mask=False,normalize=True):
    
    img = copy.deepcopy(input_mat)
    
    if normalize:
        img = mat_normalize(img)
    cmap = getattr(cm, cmap_name)
    img = cmap(img)
    
    if is_mask:
        img[:,:,3] = img[:,:,3]*input_mat
        
    img = Image.fromarray(np.uint8(img*255))
    
    return img

def crop_by_atten(lps_mat,cam_mat,atten_area):
    
    origin_lps = copy.deepcopy(np.flipud(lps_mat.T))
    origin_cam = copy.deepcopy(np.flipud(cam_mat.T))
    
    atten_lps = mat_crop(origin_lps,~atten_area,float("-inf"))
    atten_cam = mat_crop(origin_cam,~atten_area,0)
    
    return atten_lps,atten_cam

def get_atten_img(lps_mat,cam_mat,extract_lps,extract_cam,atten_bound):  
    
    img_lps=trans_to_img(np.flipud(lps_mat.T),'gist_heat',is_mask=False,normalize=True)
    atten_lps=trans_to_img(extract_lps,'gist_heat',is_mask=False,normalize=True)
    
    img_cam=trans_to_img(np.flipud(cam_mat.T),'jet',is_mask=False,normalize=True)
    atten_cam=trans_to_img(extract_cam,'jet',is_mask=False,normalize=True)
    bound_mask=trans_to_img(atten_bound,'cool',is_mask=True,normalize=False)
    
    overlay_lps = copy.deepcopy(img_lps)
    overlay_lps.paste(bound_mask,(0,0),mask=bound_mask)
    
    Img_result = {'img_lps':img_lps,'atten_lps':atten_lps,
                  'img_cam':img_cam,'atten_cam':atten_cam,
                  'bound_mask':bound_mask,'overlay_lps':overlay_lps,
                  'item_type':'image'}

    return Img_result


def save_result(input_dict,file_path,save_path):
    
    file_dir = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path).split('.')[0]
    if input_dict['item_type'].lower()=='waveform': 
        out_path = os.path.join(save_path,file_dir,file_name,'wav')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        origin_wav = utils.lps_rebuild(input_dict['origin_lps'],input_dict['origin_info'])
        extract_wav = utils.lps_rebuild(np.flipud(input_dict['atten_lps']),input_dict['origin_info'])
        
        origin_save_name=os.path.join(out_path,(file_name+'_origin.wav'))
        extract_save_name=os.path.join(out_path,(file_name+'_extract.wav'))
        
        sf.write(origin_save_name, origin_wav[:], input_dict['sampling_rate'])
        sf.write(extract_save_name, extract_wav[:], input_dict['sampling_rate'])
        
    elif input_dict['item_type'].lower()=='image':
        out_path = os.path.join(save_path,file_dir,file_name,'img')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        for img_key in input_dict.keys():
            if img_key.lower() != 'item_type':
                img_save_name=os.path.join(out_path,(file_name+'_'+img_key+'.png'))
                input_dict[img_key].save(img_save_name)
        return img_save_name
    else:
        raise KeyError('The item_type should be should belong to image or waveform')
        
    return print(input_dict['item_type']+' save finish!')

    
if __name__ == '__main__':

    model = r'C:\Users\chihk\projects\apuiv2\use_data\test_model'
    # test = r'C:\Users\chihk\projects\apuiv2\use_data\test_model\training\testwav'
    test = r'C:/Users/chihk/projects/apuiv2/recording1.wav'
    save_path = r'C:\save_data'
    test = HeatMap(model,test,save_path)
    test.run()    
