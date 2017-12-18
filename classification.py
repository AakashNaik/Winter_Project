import numpy as np
import pywt
import matplotlib.pyplot as plt

class ex_d:
	def __init__(self,data_file,time_file,l,data_repetation,reduction,sub_sets):
		self.data_file = data_file
		self.time_file = time_file
		self.l = l
		self.r = reduction
		self.dr = data_repetation
		r = reduction
		d = open(self.data_file,'r')
		t = open(self.time_file,'r')
		dr = data_repetation
		dp = []
		data = d.read().split('\n')
		ti = t.read().split('\n')
		time=[]
		labels = []
		for i in range(l):
			time.append(int(ti[i].split('\t')[0]))
			labels.append(int(ti[i].split('\t')[1]))
		self.labels = labels
		for j in range(len(time)):
			dp.append([])
			for i in range(int(r/2),dr-int(r/2)):
				dp[j].append(data[time[j]-1+i].split('\t'))
		dp2 = []
		self.d2 = dp2
		for i in range(len(dp)):
			dp2.append([])
			for j in range(len(dp[i])):
				dp2[i].append(list(map(float, dp[i][j])))
		for i in dp2:
			for j in range(len(i)):
				i[j] = np.array(i[j])
		'''
		for g in range(10):
			x = np.array(range(300))
			y = np.zeros((300))
			j=0
			for i in dp2[g]:
				y[j]+=(i[4])
				j+=1
			plt.plot(x, y)
			plt.title(labels[g])
			plt.show()
		'''

	def MEAN(self):
		dr = self.dr
		dp2 = self.d2
		l = self.l
		r = self.r
		data_point = np.zeros((l,118))
		for i in range(len(dp2)):
			for j in range(dr-r):
				data_point[i] += abs(dp2[i][j])
			data_point[i] = data_point[i]/(dr-r)
		return data_point
	def Labels(self):
		return self.labels
	def MEDIAN(self):
		dr = self.dr
		dp2 = self.d2
		l = self.l
		data_point = np.zeros((l,118))
		for i in range(len(dp2)):
			data_point[i] += dp2[i][dr//2]
		return data_point

	def VARIANCE(self):
		dr = self.dr
		dp2 = self.d2
		l = self.l
		r = self.r
		data_point = np.zeros((l,118))
		mean_points=self.MEAN()
		for i in range(len(dp2)):
			for j in range(dr-r):
				data_point[i] += abs(dp2[i][j]-mean_points[i])
			data_point[i] = data_point[i]/(dr-r)
		return data_point
	def coeff_var(self):
		return (self.VARIANCE()/self.MEAN())
#d=168*300*118
	def wavelet_features(self,z):
		d = self.d2
		dr = self.dr
		print dr
		for i in range(len(d)):
			for j in d[i]:
				j = np.array(j)
			d[i]=np.array(d[i])
		dp = np.array(d)
		r = self.r
		l = self.l
		epoch = np.zeros((118,dr))
		for i in range(118):
			for j in range(dr):
				epoch[i][j] = dp[z][j][i]

		channels=118
		cA_values=[]
		cD_values=[]
		cA_mean=[]
		cA_std = []
		cA_Energy =[]
		cD_mean = []
		cD_std = []
		cD_Energy = []
		Entropy_D = []
		Entropy_A = []
		wfeatures = []
		for i in range(channels):
			cA,cD=pywt.dwt(epoch[i,:],'coif1')
			cA_values.append(cA)
			cD_values.append(cD)		#calculating the coefficients of wavelet transform.
		for x in range(channels):
			cA_mean.append(np.mean(cA_values[x]))
			wfeatures.append(np.mean(cA_values[x]))
			cA_std.append(abs(np.std(cA_values[x])))
			wfeatures.append(abs(np.std(cA_values[x])))
			cA_Energy.append(abs(np.sum(np.square(cA_values[x]))))
			wfeatures.append(abs(np.sum(np.square(cA_values[x]))))
			cD_mean.append(np.mean(cD_values[x]))		# mean and standard deviation values of coefficents of each channel is stored .
			wfeatures.append(np.mean(cD_values[x]))

			cD_std.append(abs(np.std(cD_values[x])))
			wfeatures.append(abs(np.std(cD_values[x])))

			cD_Energy.append(abs(np.sum(np.square(cD_values[x]))))
			wfeatures.append(abs(np.sum(np.square(cD_values[x]))))

			Entropy_D.append(abs(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))))
			wfeatures.append(abs(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))))

			Entropy_A.append(abs(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x])))))
		 	wfeatures.append(abs(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x])))))
		return wfeatures
	def wave_let(self):
		d = self.d2#d2 is data 168*300*118
		dr = self.dr
		r = self.r
		l = self.l
		dp =	 np.array([self.wavelet_features(0)])
		for i in range(1,len(d)):
			dp = np.vstack([dp, self.wavelet_features(i)])
		return dp
