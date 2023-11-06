#Function to get data for melspectrogram based on directory

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from time import time
from itertools import chain

def getdata(dit,max_time_steps=109):
    
    os.chdir(dit)
    directory=os.fsencode(dit)
    data=[]; counter=0; totaltime=0; imp=[]; logs={}; filelist=[]

    #sp_cent=[]; sp_bw=[]; sp_roll=[]; mfcc=[]
    
    for file in os.listdir(directory):
        
        start=time()
        filename=os.fsdecode(file)
        filelist.append(filename)
        
        #if filename.endswith(".wav"): continue
        #print(os.path.join(directory, filename))
        #else: continue
        #scale_file=filename
        #ipd.Audio(filename)
        
        scale,sr=librosa.load(filename)
        imp.append(scale)

        #plt.figure(figsize=(20,5))
        #librosa.display.waveshow(scale,sr=sr)
        #plt.title("Waveplot",fontdict=dict(size=18))
        #plt.xlabel("Time",fontdict=dict(size=15))
        #plt.ylabel("Amplitude",fontdict=dict(size=15))
        #filter_banks=librosa.filters.mel(n_fft=2048,sr=sr,n_mels=10)
        #filter_banks.shape
        
        #Extracting Melspectrogram
        #sgram=librosa.stft(scale)
        #sgram_mag,_=librosa.magphase(sgram)
        mel_spectrogram=librosa.feature.melspectrogram(y=scale,sr=sr,n_fft=1024,hop_length=256,n_mels=10) #n_fft=1024
        #mel_spectrogram=librosa.feature.melspectrogram(y=scale,sr=sr,n_fft=100,hop_length=256,n_mels=10) #n_fft=1024

        #Feature Extraction
        #spectral_centroid=librosa.feature.spectral_centroid(y=scale,sr=sr); sp_cent.append(spectral_centroid)
        #spectral_bandwidth=librosa.feature.spectral_bandwidth(y=scale,sr=sr); sp_bw.append(spectral_bandwidth)
        #spectral_contrast=librosa.feature.spectral_contrast(y=scale,sr=sr)
        #spectral_rolloff=librosa.feature.spectral_rolloff(y=scale,sr=sr); sp_roll.append(spectral_rolloff)
        #spectral_flatness=librosa.feature.spectral_flatness(y=scale)
        #mfccs=librosa.feature.mfcc(y=scale,sr=sr); mfcc.append(mfccs)
        #mel_spectrogram.shape
        mel_spectrogram=librosa.power_to_db(mel_spectrogram,ref=np.max)

        #Ensure all spectrograms have the same width (time steps)
        if mel_spectrogram.shape[1]<max_time_steps:
            mel_spectrogram=np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:,:max_time_steps]

        data.append(mel_spectrogram)
        end=time(); duration=end-start; totaltime+=duration
        
        #Melspectrogram Image Exports
        #librosa.display.specshow(log_mel_spectrogram,x_axis="time",y_axis="mel",sr=22050)
        #plt.colorbar(format="%+2.f")
        #plt.savefig(met+filename+'.png',bbox_inches='tight',pad_inches=0,dpi=200)
        
        counter+=1; logs[counter]=duration
    
    mydict=pd.Series({'data':np.array(data),'audio_count':counter,'time':totaltime,'waves':imp,'files':filelist})
                      #'features':pd.Series({'centroid':sp_cent,'bandwidth':sp_bw,
                                            #'rolloff':sp_roll,'MFCC':mfcc})
    
    #Print data, no. of audio clips and total time to convert to melspectrograms
    print(f'No. of audio clips : {counter}, Time taken : {totaltime} seconds')
    
    return mydict

def prepdata(real,fake):
    count1,count2=real.audio_count,fake.audio_count
    real_train=np.zeros(count1)
    fake_train=np.ones(count2)
    totaud=count1+count2
    X=[] #X data
    for i in real.data: X.append(i)
    for i in fake.data: X.append(i)
    y=np.zeros(totaud) #y data
    for i in range(count1): y[i]=real_train[i]
    for i in range(count2): y[i+count1]=fake_train[i]
    X=np.array(X) #Convert X data to array
    mydict=pd.Series({'X':X,'y':y})
    return mydict

def prepfeat(real,fake):
    count1,count2=real.audio_count,fake.audio_count
    real_train=np.zeros(count1)
    fake_train=np.ones(count2)
    totaud=count1+count2
    X=[] #X data
    for i in real.features: X.append(i)
    for i in fake.features: X.append(i)
    y=np.zeros(totaud) #y data
    for i in range(count1): y[i]=real_train[i]
    for i in range(count2): y[i+count1]=fake_train[i]
    X=np.array(X) #Convert X data to array
    mydict=pd.Series({'X':X,'y':y})
    return mydict

def files_name(real,fake):
    names=chain(real.files,fake.files)
    labels=chain(np.ones(len(real.files)).tolist(),np.zeros(len(real.files)).tolist())
    dataset=chain(real.data.tolist(),fake.data.tolist())
    df=pd.DataFrame({'file':names,'melspectrogram':dataset,'label':labels})
    return df