#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pyaudio
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt

for i in range(1, 41):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "Sonidos/MI/" + str(i) + ".wav"
 
    audio = pyaudio.PyAudio()
 
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print ("recording...", i)
    frames = []
 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording", i)
    
     # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


# In[3]:


import IPython.display as ipd


# In[116]:


ipd.Audio("Sonidos/RE/1.wav")


# In[114]:


from scipy import fft, arange
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import pyaudio
import wave
from sklearn.linear_model import LogisticRegression


def frequency_sepectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)



listadoNotas = os.listdir("Sonidos/")
listadoRecords = os.listdir("Sonidos/")
valores = []
notas = []
contador = 0

for j in (listadoNotas):   
    if (j!='.ipynb_checkpoints' and j!='MEMORIA'):      
        print(j)
        contador = (contador + 1)     
        listadoRecords = os.listdir("Sonidos/" + j)
        for i in range(1, (len(listadoRecords))):
            
            # Sine sample with a frequency of 1hz and add some noise
            sr = 32  # sampling rate
            y = np.linspace(0, 2*np.pi, sr)
            y = np.tile(np.sin(y), 5)
            y += np.random.normal(0, 1, y.shape)
            t = np.arange(len(y)) / float(sr)

            frq, X = frequency_sepectrum(y, sr)

            # wav sample from https://freewavesamples.com/files/Alesis-Sanctuary-QCard-Crickets.wav
            sr, signal = wavfile.read("Sonidos/" + j + "/" + str(i) + ".wav")

            y = signal[:, 0]  # use the first channel (or take their average, alternatively)
            t = np.arange(len(y)) / float(sr)

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t, y)
            plt.xlabel('t')
            plt.ylabel('y')
            
            frq, X = frequency_sepectrum(y, sr)  
            
            plt.subplot(2, 1, 2)
            plt.plot(frq, X, 'b')
            plt.xlabel('Freq (Hz)')
            plt.ylabel('|X(freq)|')
            plt.tight_layout()

            plt.show()
            
            valores.append([max(X)])
            notas.append([contador * 100])         
            
valoresnp = np.array(valores)
notasnp = np.array(notas)

print(valores)
print(notas)

from sklearn.linear_model import LogisticRegression

regresion_logistica = LogisticRegression()
regresion_logistica.fit(valoresnp,notasnp.ravel())

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "Sonidos/MEMORIA/sonido.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            frames_per_buffer=CHUNK)
print ("recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print ("finished recording")

 # stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

sr = 32  # sampling rate
y = np.linspace(0, 2*np.pi, sr)
y = np.tile(np.sin(y), 5)
y += np.random.normal(0, 1, y.shape)
t = np.arange(len(y)) / float(sr)

frq, X = frequency_sepectrum(y, sr)

sr, signal = wavfile.read("Sonidos/MEMORIA/sonido.wav")

y = signal[:, 0]  # use the first channel (or take their average, alternatively)
t = np.arange(len(y)) / float(sr)

frq, X = frequency_sepectrum(y, sr)
print(max(X))
X_nuevo = np.array([[max(X)]])
prediccion = regresion_logistica.predict(X_nuevo)
print(prediccion)

score = regresion_logistica.score(valoresnp, notasnp)
print(score)


# In[135]:


import os
arr = os.listdir("Sonidos/")
print(arr)


# In[133]:


import numpy as np
x = []
x.append([0.56, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
x.append([0.50, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
X = np.array(x, "float32")
y = np.array([[0],[1]], "float32")

from sklearn.linear_model import LogisticRegression

print(x)
regresion_logistica = LogisticRegression()

regresion_logistica.fit(X,y.ravel())

X_nuevo = np.array([[0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5]])

prediccion = regresion_logistica.predict(X_nuevo)
print(prediccion)

