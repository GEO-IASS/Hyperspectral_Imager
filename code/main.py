import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.fftpack import dct
from scipy.sparse import linalg
from scipy.ndimage.interpolation import zoom

# All based on work from http://www.scielo.org.co/pdf/iei/v34n3/v34n3a09.pdf

N0 = 1160
M0 = 320
L0 = 360

zfactor = 0.25
offset = 1

N = int(N0*zfactor)
M = int(M0*zfactor)
L = int(L0*zfactor - offset)

Delta = 2

np.random.seed(0)

imdir = "../images/DeepHorizon_OilSpill/"
imname = "0612-1615_rad_sub.dat"
impath = imdir + imname

def readData(fname,N0,M0,L0,zfactor):
	data = np.fromfile(fname,dtype=np.int16)
	data = np.reshape(data,(N0,L0,M0))
	data = np.swapaxes(data,1,2)
	data = zoom(data,zfactor)
	return data[:,:,:-offset]

def genMask1Matrix(N,M,L):
	m_v = np.round(np.random.rand(N*M,))
	r = np.arange(N*M*L)
	c = np.arange(N*M*L)
	data = np.tile(m_v,L)
	T = sparse.csr_matrix((data,(r,c)),shape=(N*M*L,N*M*L))
	return T

# def genMask2Matrix(N,M,L):
# 	data = np.round(np.random.rand(N*(M+L-1),))
# 	r = np.arange(N*(M+L-1))
# 	c = np.arange(N*(M+L-1))
# 	T = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1),N*(M+L-1)))
# 	return T

def genMask2Matrix(N,M,L):
	data = np.ones((N*(M+L-1),))
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
			c[i*N*Delta + j*Delta**2:i*N*Delta + j*Delta**2 + Delta**2] = c0 + j*Delta + i*Delta**2
	D = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1)/Delta**2,N*(M+L-1)))
	return D

def dct3(x,N,M,L):
	x = np.reshape(x,(N,M,L))
	X = dct(dct(dct(x).transpose(0,2,1)).transpose(1,2,0)).transpose(1,2,0).transpose(0,2,1)
	X = np.reshape(X,(N*M*L,))
	return X

def idct3(X,N,M,L):
	X = np.reshape(X,(N,M,L))
	x = dct(dct(dct(X).transpose(1,2,0)).transpose(0,2,1)).transpose(0,2,1).transpose(1,2,0)
	x = np.reshape(x,(N*M*L,))
	return x

def A_forward(theta,X,N,M,L):
	f = idct3(theta,N,M,L)
	g = X.dot(f)
	return g

def A_backward(g,X,N,M,L):
	Xt = X.T
	f = Xt.dot(g)
	theta = dct3(f,N,M,L)
	return theta

data = readData(impath,N0,M0,L0,zfactor)

T1 = genMask1Matrix(N,M,L)
T2 = genMask2Matrix(N,M,L)
P = genDispMatrix(N,M,L)
# D = genDecimationMatrix(N,M,L,Delta)
H = T2.dot(P.dot(T1))

f_img = np.reshape(data,(N*M*L,1))

X = H
A = linalg.LinearOperator(X.shape,matvec = lambda theta: A_forward(theta,X,N,M,L),rmatvec = lambda g: A_backward(g,X,N,M,L))
theta0 = dct3(f_img,N,M,L)
g = A.dot(theta0)

# im = np.reshape(g,(int(N/Delta),int((M+L-1)/Delta)))
im = np.reshape(g,(N,(M+L-1)))

result = linalg.lsqr(A,g,damp=0.1,show=1,iter_lim=500)

theta = result[0]

f_v = idct3(theta,N,M,L)
f = np.reshape(f_v,(N,M,L))

img = im.mean(axis=2)

plt.imshow(img,cmap='Greys_r')
plt.show()
plt.imshow(data.mean(axis=2),cmap='Greys_r')
plt.show()