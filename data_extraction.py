# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.

#this line above prepares IPython notebook for working with matplotlib

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().

import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns #sets up styles and gives us more plotting options
#import DataFrame as df
# Import all libraries needed for the tutorial

# General syntax to import specific functions in a library:
##from (library) import (specific library function)
from pandas import DataFrame, read_csv

# General syntax to import a library but no functions:
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
import re
import scipy as sc

path= r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_cnt.txt'
data=pd.read_csv(path,header=None)
path2=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_mrk.txt'
mark=pd.read_csv(path2,header=None)



# converted data txt file in  2-d array
data_array=[]
for r in range(0,data.shape[0]-1):
    data_array.append(map(float,data.iloc[r,0].split('\t')))
#converted mark txt file in 2-d array
mark_array=[]
for d in range(0,mark.shape[0]-1):
    mark_array.append(map(float,mark.iloc[d,0].split('\t')))

array=[]
for r in range(1,119):
    array.append(r)
# 2-d array mark_array  converted in to data frame
marker_data=pd.DataFrame(mark_array,columns=['Time','Number'])
data_value=pd.DataFrame(data_array)
data_value.columns=array


mean=[]
for r in range (0,marker_data.shape[0]-1):
    x= marker_data.iloc[r,0]
    mean.append([])
    for f in range(0,118):
        summ=0
        for d in range(int(x),int(x)+300):
            summ+=data_value.iloc[d,f]
        mean[r].append(summ/300.0)

mean_data=pd.DataFrame(mean)
mean_data.columns=array
mean_data.to_csv(r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/mean.txt', header=None, index=None, sep=' ', mode='w')

print data_value









#s=pd.DataFrame(s.str.split('\t'))
#np.array(s.split('\n'))

'''s=s["data"]
s.str.split("\n").head(8)
r=s.shape[0]
print type(s)
s=s.values
t=s.tostring()
type(t)
r=s.shape[0]
s=np.array2string(s)

print t[16]
array=[]

for p in range(0,r-1):

    array.append( np.char.split(t[r],sep='\t'))

    #array.append( np.array(list(s[r]), dtype=float,split='\t'))
print array[2]
'''

'''array=[]
for r in range(0,s.shape()-1):
    array.append()
    b=map(float,b)
    print b
'''
