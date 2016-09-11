import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.fftpack import dct
from sklearn import linear_model
from scipy.ndimage.interpolation import zoom

# All based on work from http://www.scielo.org.co/pdf/iei/v34n3/v34n3a09.pdf

N0 = 1160
M0 = 320
L0 = 360

zfactor = 0.125

N = int(N0*zfactor)
M = int(M0*zfactor)
L = int(L0*zfactor - 1)

Delta = 4

np.random.seed(0)

imdir = "../images/DeepHorizon_OilSpill/"
imname = "0612-1615_rad_sub.dat"
impath = imdir + imname

def readData(fname,N0,M0,L0,zfactor):
	data = np.fromfile(fname,dtype=np.int16)
	data = np.reshape(data,(N0,L0,M0))
	data = np.swapaxes(data,1,2)
	data = zoom(data,zfactor)
	return data[:,:,:-1]

def genMask1Matrix(N,M,L):
	m_v = np.round(np.random.rand(N*M,))
	r = np.arange(N*M*L)
	c = np.arange(N*M*L)
	data = np.tile(m_v,L)
	T = sparse.csr_matrix((data,(r,c)),shape=(N*M*L,N*M*L))
	return T

def genMask2Matrix(N,M,L):
	data = np.round(np.random.rand(N*(M+L-1),))
	r = np.arange(N*(M+L-1))
	c = np.arange(N*(M+L-1))
	T = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1),N*(M+L-1)))
	return T

def genDispMatrix(N,M,L):
	data = np.ones((N*M*L,))
	c = np.arange(N*M*L)
	r0 = np.arange(N*M)
	r = np.empty(N*M*L)
	for i in range(L):
		r[i*N*M : (i+1)*N*M] = r0+i*N
	P = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1),N*M*L))
	return P

def genDecimationMatrix(N,M,L,Delta):
	data0 = np.ones((Delta**2,))
	r0 = np.zeros((Delta**2,))
	c0 = np.arange(Delta**2)
	data = np.ones((N*(M+L-1),))
	r = np.empty((N*(M+L-1),))
	c = np.empty((N*(M+L-1),))
	for i in range(int((M+L-1)/Delta)):
		for j in range(int(N/Delta)):
			r[i*N*Delta + j*Delta**2:i*N*Delta + j*Delta**2 + Delta**2] = r0 + i*N/Delta + j
			c[i*N*Delta + j*Delta**2:i*N*Delta + j*Delta**2 + Delta**2] = np.roll(c0,j*Delta) + i*Delta**2
	D = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1)/Delta**2,N*(M+L-1)))
	return D

data = readData(impath,N0,M0,L0,zfactor)

T1 = genMask1Matrix(N,M,L)
T2 = genMask2Matrix(N,M,L)
P = genDispMatrix(N,M,L)
D = genDecimationMatrix(N,M,L,Delta)
H = T2.dot(P.dot(T1))

f_img = np.reshape(data,(N*M*L,1))

X = D.dot(H)
y = X.dot(f_img)
f0 = np.zeros((N*M*L,))

clf = linear_model.Lasso(alpha = 0.1)
clf.fit(X,y)

f_v = clf.coef_
f = np.reshape(f_v,(N,M,L))

img = np.log(f.mean(axis=2))

plt.imshow(img,cmap='Greys_r')
plt.show()