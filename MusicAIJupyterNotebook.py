#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello World!")


# In[2]:


import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
from IPython.display import Audio
from scipy import stats


# In[3]:


y, sr = librosa.load("yours.mp3")


# In[4]:


# loading the audio file
# sphinx_gallery_thumbnail_path = '_static/playback-thumbnail.png'

# We'll need IPython.display's Audio widget


# In[5]:


Audio(data=y, rate=sr)


# In[6]:


print(y)


# In[7]:


# estimating the tempo of the song
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
tempo


# In[8]:


# class that uses the librosa library to analyze the key that an mp3 is in
# arguments:
#     waveform: an mp3 file loaded by librosa, ideally separated out from any percussive sources
#     sr: sampling rate of the mp3, which can be obtained when the file is read with librosa
#     tstart and tend: the range in seconds of the file to be analyzed; default to the beginning and end of file if not specified
class Tonal_Fragment(object):
    def __init__(self, waveform, sr, tstart=None, tend=None):
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend
        
        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        self.y_segment = self.waveform[self.tstart:self.tend]
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, bins_per_octave=24)
        
        # chroma_vals is the amount of each pitch class present in this time interval
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        # dictionary relating pitch names to the associated intensity in the song
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)} 
        
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
        # data above to typical profiles of major and minor keys:
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # finds correlations between the amount of each pitch class in the time interval and the above profiles,
        # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
        self.min_key_corrs = []
        self.maj_key_corrs = []
        for i in range(12):
            key_test = [self.keyfreqs.get(pitches[(i + m)%12]) for m in range(12)]
            # correlation coefficients (strengths of correlation for each key)
            self.maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1,0], 3))
            self.min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1,0], 3))
            
        # names of all major and minor keys
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)}, 
                         **{keys[i+12]: self.min_key_corrs[i] for i in range(12)}}
        
        # this attribute represents the key determined by the algorithm
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())
        
        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr*0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr
                
    # prints the relative prominence of each pitch class            
    def print_chroma(self):
        self.chroma_max = max(self.chroma_vals)
        for key, chrom in self.keyfreqs.items():
            print(key, '\t', f'{chrom/self.chroma_max:5.3f}')
                
    # prints the correlation coefficients associated with each major/minor key
    def corr_table(self):
        for key, corr in self.key_dict.items():
            print(key, '\t', f'{corr:6.3f}')
    
    def correlation_coefficients(self):
        return self.key_dict.items()
    
    # printout of the key determined by the algorithm; if another key is close, that key is mentioned
    def print_key(self):
        print("likely key: ", max(self.key_dict, key=self.key_dict.get), ", correlation: ", self.bestcorr, sep='')
        if self.altkey is not None:
                print("also possible: ", self.altkey, ", correlation: ", self.altbestcorr, sep='')
    
    # prints a chromagram of the file, showing the intensity of each pitch class over time
    def chromagram(self, title=None):
        C = librosa.feature.chroma_cqt(y=self.waveform, sr=sr, bins_per_octave=24)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        if title is None:
            plt.title('Chromagram')
        else:
            plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.show()


# In[9]:


# this audio takes a long time to load because it has a very high sampling rate; be patient.
# the load function generates a tuple consisting of an audio object y and its sampling rate sr
# this function filters out the harmonic part of the sound file from the percussive part, allowing for
# more accurate harmonic analysis
y_harmonic, y_percussive = librosa.effects.hpss(y)


# In[10]:


# this block instantiates the Tonal_Fragment class with the first 22 seconds of the above harmonic part of une barque.
# the three methods called will print the determined key of the song, the correlation coefficients for all keys,
# and a chromogram, which shows the intensity of frequencies associated with each of the 12 pitch classes over time.

unebarque_fsharp_min = Tonal_Fragment(y_harmonic, sr, tend=22)
unebarque_fsharp_min.print_chroma()


# In[11]:


unebarque_fsharp_min.print_key()
unebarque_fsharp_min.corr_table()
unebarque_fsharp_min.chromagram("Une Barque sur l\'Ocean")


# In[12]:


# parts of the song that are more tonally ambiguous will show two keys with print_key(),
# if they are similarly well-correlated.
# this section of une barque is in E minor, though the algorithm suggests that it is in D major, a closely related key,
# though E minor is also listed since their correlation coefficients are very close.
unebarque_e_min = Tonal_Fragment(y_harmonic, sr, tstart=0)
unebarque_e_min.print_key()
unebarque_e_min.corr_table()


# In[13]:


# in the case of une barque sur l'ocean (and other songs), predictions become less reliable over short time frames
# the below block prints the predicted key of every 3-second-long cut of the piece.
bin_size = 3
for i in range(24):
    fragment = Tonal_Fragment(y_harmonic, sr, tstart = bin_size*i, tend=bin_size*(i+1))
    print(bin_size*i,"sec:",fragment.key)
    if fragment.altkey is not None:
        print("\t or:", fragment.altkey)


# In[14]:


# getting volume graph
rms = librosa.feature.rms(y=y)
plt.plot(rms[0])
plt.show()


# In[15]:


print(rms)
print(np.mean(rms[0]))
print(np.max(rms[0]))


# In[16]:


import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
from IPython.display import Audio
from scipy import stats
# Python program to explain os.listdir() method 
    
# importing os module 
import os
  
# Get the path of current working directory
path = os.getcwd()
path += "//Good Songs"
  
# Get the list of all files and directories
# in current working directory
songs_list = os.listdir(path)

#move the non-song item to end of list so that we can avoid using it for the data
songs_list.sort(key = 'librosapractice.ipynb'.__eq__)

# print the list
print(songs_list)


for song in songs_list:
    if song[-3:] == 'mp3':
        print(song)
        # load audio
        y, sr = librosa.load(path+"//"+song)
        Audio(data=y, rate=sr)
        print(y)
        # tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        print(tempo)
        # key
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo


        y_harmonic, y_percussive = librosa.effects.hpss(y)

        song_tonal_analysis = Tonal_Fragment(y_harmonic, sr, tend=22)
        song_tonal_analysis.print_chroma()
        song_tonal_analysis.print_key()
        song_tonal_analysis.corr_table()
        song_tonal_analysis.chromagram("song")

        unebarque_e_min = Tonal_Fragment(y_harmonic, sr, tstart=0)
unebarque_e_min.print_key()
unebarque_e_min.corr_table()


# In[17]:


# for extracting likely key/s for each song + spectogram
for song in songs_list:
    if song[-3:] == 'mp3':
        print(song)
        y, sr = librosa.load(song)
        Audio(data=y, rate=sr)
        print(y)
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    tempo
          
            
    y_harmonic, y_percussive = librosa.effects.hpss(y)
        
    song_tonal_analysis = Tonal_Fragment(y_harmonic, sr, tend=22)
    song_tonal_analysis.print_chroma()
    song_tonal_analysis.print_key()
    song_tonal_analysis.corr_table()
    song_tonal_analysis.chromagram("song")
    
    unebarque_e_min = Tonal_Fragment(y_harmonic, sr, tstart=0)
unebarque_e_min.print_key()
unebarque_e_min.corr_table()


# In[18]:


# getting volume graph for each song in the list
def extract_volume(song):
    y, sr = librosa.load(song)
    Audio(data=y, rate=sr)
    rms = librosa.feature.rms(y=y)
    # plt.plot(rms[0])
    # plt.show()
    # print(rms)
    # print("volume mean: " + str(np.mean(rms[0])))
    # print("volume max: " + str(np.max(rms[0])))
    return np.mean(rms[0]), np.max(rms[0])


# In[19]:


# tempo function
def extract_tempo(song):
    y, sr = librosa.load(song)
    Audio(data=y, rate=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    # print("tempo: " + str(tempo))
    return tempo


# In[20]:


def extract_key(song):
    y, sr = librosa.load(song)
    Audio(data=y, rate=sr)
    # print(y)
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
          
            
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    """
    song_tonal_analysis = Tonal_Fragment(y_harmonic, sr, tend=22)
    song_tonal_analysis.print_chroma()
    song_tonal_analysis.print_key()
    song_tonal_analysis.corr_table()
    song_tonal_analysis.chromagram("song")
    """    
    
    keyanalysis = Tonal_Fragment(y_harmonic, sr, tstart=0)
    # keyanalysis.print_key()
    # unebarque_e_min.corr_table()
    return keyanalysis.correlation_coefficients()


# In[21]:


list_of_emotions = []
list_of_all_songs = []
for song in songs_list:
    if song[-3:] == 'mp3':
        #print(song)
        tempo = extract_tempo(song)
        volume = extract_volume(song)
        key = extract_key(song)
        key_probabilities = [x[1] for x in list(key)]
        combined_list = [tempo[0]] + [volume[0], volume[1]] + key_probabilities
        #print(combined_list)
        list_of_all_songs += [combined_list]
    else:
        print(song)
        # I think I need to add the below code here???
print(list_of_all_songs)


# In[22]:


graph_colors = ['red', 'cyan', 'hotpink', 'lightslategrey']
from sklearn.decomposition import PCA
import csv
emotions_list = []
 
# opening the CSV file
#with open('final_music_dataset - Sheet1 (1).csv', mode ='r')as file:
with open('final_music_dataset - Sheet1 (1).csv', mode ='r')as file:
   
  # reading the CSV file
  csvFile = csv.reader(file)
 
  # displaying the contents of the CSV file
  n = 0
  for line in csvFile:
        # print(lines)
        if n >= 1:
            if line[3] == 'calm/neutral':
                emotions_list += [graph_colors[1]] 
            elif line[3] == 'happy/joyous':
                emotions_list += [graph_colors[2]] 
            elif line[3] == 'sad/melancholy':
                emotions_list += [graph_colors[3]] 
            elif line[3] == 'angry/restlessness':
                emotions_list += [graph_colors[0]] 
            else:
                print(line[3])
        n += 1
        # print(line[3] == "calm/neutral")
        
pca = PCA(n_components=2)
list_of_songs_numpy = np.array(list_of_all_songs)
print(list_of_songs_numpy)
two_d_output = pca.fit_transform(list_of_songs_numpy)
print(two_d_output)
print(emotions_list)


# In[23]:


import matplotlib.pyplot as plt
plt.scatter([v[0] for v in two_d_output], [v[1] for v in two_d_output], c = emotions_list)
plt.show()


# In[24]:


#import csv
 
# opening the CSV file
#with open('Music Dataset - Sheet1.csv', mode ='r')as file:
   
  # reading the CSV file
 # csvFile = csv.reader(file)
 
  # displaying the contents of the CSV file
  #for lines in csvFile:
       # print(lines)
        #print(lines[3])


# In[ ]:





# In[ ]:




