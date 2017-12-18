import numpy as np
import pyedflib
from matplotlib import pyplot as plt
from nitime import utils
from nitime import algorithms as alg
from nitime.timeseries import TimeSeries
from nitime.viz import plot_tseries
import csv
import pywt
import scipy.stats as sp
from scipy import signal
#from spectrum import *
from os import listdir
from os.path import isfile, join
from wyrm import processing as proc
from wyrm.types import Data
from wyrm.io import convert_mushu_data
from sklearn import metrics
from wyrm.processing import calculate_csp,segment_dat,apply_csp,append_epo
from wyrm.processing import select_channels
from wyrm.processing import swapaxes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.cluster import MiniBatchKMeans,KMeans,SpectralClustering,MeanShift,AffinityPropagation,AgglomerativeClustering,DBSCAN,Birch,FeatureAgglomeration
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import pandas as pd

data_file = r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_cnt.txt'
marker_file = r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_mrk.txt'

channels = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6',
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4',
                'FFC6', 'FFC8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC7',
                'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'
                , 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8',
                'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1', 'PPO2', 'PPO6',
                'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1', 'Oz', 'O2', 'OI1', 'OI2',
                'I1', 'I2']



signal_array = np.loadtxt(data_file)
b, a = signal.butter(3, np.array([7, 30])/ 100.00, 'bandpass')
signal_array1 = signal.lfilter(b, a, signal_array, axis = -1)

marker_array = [list(map(str,l.split())) for l in open(marker_file).readlines()]
time = np.arange(len(signal_array))
train_markers1 = [(float(events[0]),str(events[1])) for events in marker_array if int(events[1])!= 0]
for events in marker_array:
    if int(events[1]) != 0:
        train_markers1.append((float(events[0]) + 100.0, str(events[1])))
        train_markers1.append((float(events[0]) + 200.0, str(events[1])))

markers1 = np.array(train_markers1)
y1 = [0] * len(markers1)

markers_subject1_class_1 = [(float(events[0]),str(events[1])) for events in markers1 if events[1] == '1']
markers_subject1_class_2 = [(float(events[0]),str(events[1])) for events in markers1 if events[1] == '2']

cnt1 = convert_mushu_data(signal_array, markers_subject1_class_1,100,channels)
cnt2 = convert_mushu_data(signal_array, markers_subject1_class_2,100,channels)

md = {'class 1': ['1'],'class 2': ['2']}

epoch_subject1_class1 = segment_dat(cnt1, md, [0, 1000])
epoch_subject1_class2 = segment_dat(cnt2, md, [0, 1000])

def bandpowers(segment):
    features = []
    for i in range(len(segment)):
        f,Psd = signal.welch(segment[i,:], 100)
        power1 = 0
        power2 = 0
        f1 = []
        for j in range(0,len(f)):
            if(f[j]>=4 and f[j]<=13):
                power1 += Psd[j]
            if(f[j]>=14 and f[j]<=30):
                power2 += Psd[j]
        features.append(power1)
        features.append(power2)
    return features

# Wavelet Features Def

def wavelet_features(epoch):
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy =[]
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    features = []
    for i in range(len(epoch)):
        cA,cD=pywt.dwt(epoch[i,:],'coif1')
        cA_values.append(cA)
        cD_values.append(cD)		#calculating the coefficients of wavelet transform.
    for x in range(len(epoch)):
        cA_Energy.append(abs(np.sum(np.square(cA_values[x]))))
        features.append(abs(np.sum(np.square(cA_values[x]))))

    for x in range(len(epoch)):
        cD_Energy.append(abs(np.sum(np.square(cD_values[x]))))
        features.append(abs(np.sum(np.square(cD_values[x]))))

    return features

final_epoch1 = append_epo(epoch_subject1_class1, epoch_subject1_class2)
w1, a1, d1 = calculate_csp(final_epoch1, [0, 1])
fil_epoch_subject1_class1 = apply_csp(epoch_subject1_class1, w1, [0,1,2,3,4,-5,-4,-3,-2,-1])
fil_epoch_subject1_class2 = apply_csp(epoch_subject1_class2, w1, [0,1,2,3,4,-5,-4,-3,-2,-1])
fil_final_epoch1 = append_epo(fil_epoch_subject1_class1, fil_epoch_subject1_class2)
print(fil_epoch_subject1_class1.data.shape, fil_epoch_subject1_class2.data.shape, fil_final_epoch1.data.shape)

# Band_power and wavelet dictionary generation


dictionary1 = []
dictionary2 = []

for i in range(len(fil_final_epoch1.axes[0])):
    segment = fil_final_epoch1.data[i]
    segment = np.array(segment)
    segment = np.transpose(segment)
    features1 = bandpowers(segment)
    features2 = wavelet_features(segment)
    dictionary1.append(features1)
    dictionary2.append(features2)


dictionary1 = np.array(dictionary1)
dictionary2 = np.array(dictionary2)
dictionary_bandpower  = dictionary1
dictionary_wavelet = dictionary2

targets = fil_final_epoch1.axes[0]
print((targets))
print(dictionary_bandpower.shape)
print(dictionary_wavelet.shape)
print(dictionary_wavelet[0])
'''
x = np.arange(504)
for j in range(20):
	y = [dictionary_wavelet[i,j] for i in np.arange(504)]
	plt.figure()
	plt.plot(x,y)
plt.show()
'''


km = KMeans(n_clusters=2,max_iter = 5000,n_init = 50)
km.fit(dictionary_wavelet)
labels = km.labels_
print(labels)

mkm = MiniBatchKMeans(n_clusters=2,max_iter = 5000,n_init = 50)
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
print(labels)

mkm = SpectralClustering(n_clusters=2)
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
print(labels)

mkm = MeanShift()
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
print(labels)

mkm = AffinityPropagation()
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
print(labels)

mkm = AgglomerativeClustering(n_clusters =2)
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
print(labels)

mkm = DBSCAN()
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
print(labels)

mkm = Birch()
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
print(labels)

km = KMeans(n_clusters=2,max_iter = 5000,n_init = 50)
km.fit(dictionary_bandpower)
labels = km.labels_
print(labels)

mkm = MiniBatchKMeans(n_clusters=2,max_iter = 5000,n_init = 50)
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
print(labels)

mkm = SpectralClustering(n_clusters=2)
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
print(labels)

mkm = MeanShift()
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
print(labels)

mkm = AffinityPropagation()
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
print(labels)

mkm = AgglomerativeClustering(n_clusters =2)
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
print(labels)

mkm = DBSCAN()
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
print(labels)

mkm = Birch()
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
print(labels)




# For Wavelet

from scipy.sparse import coo_matrix
X_sparse = coo_matrix(dictionary_wavelet)
print(X_sparse)


from sklearn.utils import resample
dictionary_wavelet, X_sparse, ywe = resample(dictionary_wavelet, X_sparse, targets, random_state=0)

# For Band-Power

from scipy.sparse import coo_matrix
X_sparse = coo_matrix(dictionary_bandpower)

from sklearn.utils import resample
dictionary_bandpower, X_sparse, ybp = resample(dictionary_bandpower, X_sparse, targets, random_state=0)

#Wavelet_dictionary

from sklearn.model_selection import KFold

dictionary = dictionary_wavelet

kf = KFold(n_splits=10,random_state = 30, shuffle = True)
kf.get_n_splits(dictionary)

y_classifier1 = []
y_classifier2 = []
y_classifier3 = []
y_classifier4 = []

y_all1 = []
y_all2 = []
y_all3 = []
y_all4 = []
y_example_test = []
y_final_test = []

print(kf)

def calculate_accuracy(Xts,yts,D,class1,class2,n):
    y_pred1 = []
    y_pred2 = []
    y_pred3 = []
    y_pred4 = []
    diff = []
    counter1 = 0
    counter2 = 0
    for i in range(len(Xts)):
        features = Xts[i]

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
        omp.fit(D,features)
        coef = omp.coef_
        p = 0
        q = 0
        l = 0
        m = 0
        a = 0
        b = 0

        list1 = coef[0:min(class1,class2)]
        list2 = coef[min(class1,class2)+1:2*min(class1,class2)]

        c1 = (sum(z*z for z in list1))**(1/2.0)
        c2 = (sum(z*z for z in list2))**(1/2.0)

        p = np.std(list1)
        q = np.std(list2)

        a = max(list1)
        b = max(list2)

        for ko in range(min(class1,class2)):
            l = l + coef[ko]

        for io in xrange(min(class1,class2)+1,2*min(class1,class2)):
            m = m + coef[io]

        if p > q:
            y_pred1.append(0)

        else:
            y_pred1.append(1)
            if(yts[i] != 0):
                if(counter1==0):
                    counter1 +=1
                    idx_r, = coef.nonzero()
                    plt.xlim(0, len(coef))
                    plt.title("Sparse Signal")
                    plt.stem(idx_r, coef[idx_r])
                    plt.show()

        if l>m:
            y_pred2.append(0)
        else:
            y_pred2.append(1)

        if a>b:
            y_pred3.append(0)
        else:
            y_pred3.append(1)

        if c1 > c2:
            y_pred4.append(0)
        else:
            y_pred4.append(1)


    print('class1', metrics.accuracy_score(yts, y_pred4, normalize=True, sample_weight=None))

    print('class2', metrics.accuracy_score(yts, y_pred3, normalize=True, sample_weight=None))

    print('class3', metrics.accuracy_score(yts, y_pred1, normalize=True, sample_weight=None))

    print('class4',metrics.accuracy_score(yts, y_pred2, normalize=True, sample_weight=None))

    print('\n')


    return y_pred4,y_pred3,y_pred1,y_pred2

for train_index, test_index in kf.split(dictionary):
    X_train, X_test = dictionary[train_index], dictionary[test_index]
    y_train, y_test = ywe[train_index], ywe[test_index]
    class1 = 0
    class2 = 0
    for i in range(len(y_train)):
        if(y_train[i] == 0):
            class1 += 1;
        else:
            class2 += 1;
    reb_y = []
    reb_dic = []
    count = 0
    iterator = 0
    while(count < min(class1,class2)):
        if y_train[iterator] == 0:
            reb_dic.append(X_train[iterator])
            reb_y.append(0)
            count += 1
        iterator += 1

    count = 0
    iterator = 0
    while(count < min(class1,class2)):
        if y_train[iterator] == 1:
            reb_dic.append(X_train[iterator])
            reb_y.append(1)
            count += 1
        iterator += 1
    reb_dictionary = np.array(reb_dic)
    reb_dictionary = reb_dictionary.transpose()
    y_classifier1,y_classifier2,y_classifier3,y_classifier4 = calculate_accuracy(X_test,y_test,reb_dictionary,class1,class2,5)

    y_all1.extend(y_classifier1);
    y_all2.extend(y_classifier2);
    y_all3.extend(y_classifier3);
    y_all4.extend(y_classifier4);

    y_final_test.extend(y_test);
    y_example_test = y_test

#Bandpower_dictionary

from sklearn.model_selection import KFold

dictionary = dictionary_bandpower

kf = KFold(n_splits=10,random_state = 30, shuffle = True)
kf.get_n_splits(dictionary)

y_classifier1 = []
y_classifier2 = []
y_classifier3 = []
y_classifier4 = []

y_all1 = []
y_all2 = []
y_all3 = []
y_all4 = []
y_example_test = []
y_final_test = []

print(kf)


for train_index, test_index in kf.split(dictionary):
    X_train, X_test = dictionary[train_index], dictionary[test_index]
    y_train, y_test = ybp[train_index], ybp[test_index]
    class1 = 0
    class2 = 0
    for i in range(len(y_train)):
        if(y_train[i] == 0):
            class1 += 1;
        else:
            class2 += 1;
    reb_y = []
    reb_dic = []
    count = 0
    iterator = 0
    while(count < min(class1,class2)):
        if y_train[iterator] == 0:
            reb_dic.append(X_train[iterator])
            reb_y.append(0)
            count += 1
        iterator += 1

    count = 0
    iterator = 0
    while(count < min(class1,class2)):
        if y_train[iterator] == 1:
            reb_dic.append(X_train[iterator])
            reb_y.append(1)
            count += 1
        iterator += 1
    reb_dictionary = np.array(reb_dic)
    reb_dictionary = reb_dictionary.transpose()
    y_classifier1,y_classifier2,y_classifier3,y_classifier4 = calculate_accuracy(X_test,y_test,reb_dictionary,class1,class2,5)

    y_all1.extend(y_classifier1);
    y_all2.extend(y_classifier2);
    y_all3.extend(y_classifier3);
    y_all4.extend(y_classifier4);

    y_final_test.extend(y_test);
    y_example_test = y_test
