import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy.sparse import linalg
from scipy.misc import imresize
from sklearn.linear_model import Lasso
from NESTA import NESTA
from spgl1 import *
from spgl_aux import * 
from scipy.ndimage.interpolation import zoom

# All based on work from http://www.scielo.org.co/pdf/iei/v34n3/v34n3a09.pdf

N0 = 1160
M0 = 320
L0 = 360

zfactor = 0.125
offset = 4
Nmax = 1024

N = int(Nmax*zfactor)
M = int(M0*zfactor)
L = int(L0*zfactor - offset)

Delta = 4

np.random.seed(0)

imdir = "../images/DeepHorizon_OilSpill/"
imname = "0612-1615_rad_sub.dat"
impath = imdir + imname

def readData(fname,N0,M0,L0,zfactor,offset,Nmax):
	data = np.fromfile(fname,dtype=np.int16)
	data = np.reshape(data,(N0,L0,M0))
	data = np.swapaxes(data,1,2)
	data = zoom(data[0:Nmax,:,:],zfactor)
	return data[:,:,:-offset]

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

# def genMask2Matrix(N,M,L):
# 	data = np.ones((N*(M+L-1),))
# 	r = np.arange(N*(M+L-1))
# 	c = np.arange(N*(M+L-1))
# 	T = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1),N*(M+L-1)))
# 	return T

def genDispMatrix(N,M,L):
	data = np.ones((N*M*L,))
	c = np.arange(N*M*L)
	r0 = np.arange(N*M)
	r = np.empty(N*M*L)
	for i in range(L):
		r[i*N*M : (i+1)*N*M] = r0+i*N
	P = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1),N*M*L))
	return P

# def genDispMatrix(N,M,L):
# 	data = np.ones((N*M*L,))
# 	c0 = np.arange(0,N*M,M)
# 	r0 = np.arange(N*M)
# 	c = np.empty(N*M*L)
# 	r = np.empty(N*M*L)
# 	for i in range(L):
# 		for j in range(M):
# 			c[i*N : (i+1)*N] = c0+i*N*M+j
# 			print(c0+i*N*M+j)
# 		r[i*N*M : (i+1)*N*M] = r0+i*N
# 	P = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1),N*M*L))
# 	return P

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
	X = dct(dct(dct(x).transpose(0,2,1)).transpose(1,2,0)).transpose(2,0,1).transpose(0,2,1)
	X = np.reshape(X,(N*M*L,))
	return X

def idct3(X,N,M,L):
	X = np.reshape(X,(N,M,L))
	x = idct(idct(idct(X).transpose(0,2,1)).transpose(1,2,0)).transpose(2,0,1).transpose(0,2,1)
	x = np.reshape(x,(N*M*L,))
	return x

def A_forward(theta,X,N,M,L,Delta):
	f = idct3(theta,N,M,L)
	g = X.dot(f)
	g_im = g.reshape((M+L-1,N))
	# plt.imshow(g_im,cmap='Greys_r',interpolation='none')
	# plt.show()
	g_d = imresize(g_im,1./Delta)
	g_v = g_d.reshape((int(N*(M+L-1)/Delta**2),))
	return g_v

def A_backward(g,X,N,M,L,Delta):
	g_d = g.reshape((int((M+L-1)/Delta),int(N/Delta)))
	g_im = zoom(g_d,Delta,order=0)
	g = g_im.reshape((N*(M+L-1),))
	Xt = X.T
	f = Xt.dot(g)
	theta = dct3(f,N,M,L)
	return theta

def Amatrix(x,mode,Af,Ab):
	if mode == 1:
		return Af(x)
	else:
		return Ab(x)

data = readData(impath,N0,M0,L0,zfactor,offset,Nmax)

# N = 20
# M = 20
# L = 15
# Delta = 2

# # l1 = np.ones((N,M))
# # data = np.zeros((N,M,L))
# data = np.ones((N,M,L))
# data[3:7,3:8,:] = np.zeros((4,5,15))
# im_data = np.random.rand(N,M+L-1)
# im_v =im_data.reshape((N*(M+L-1),))
# data = np.stack((l1,2*l1,3*l1,l1,0.5*l1,l1,0.5*l1),axis=2)

T1 = genMask1Matrix(N,M,L)
T2 = genMask2Matrix(N,M,L)
P = genDispMatrix(N,M,L)
# D = genDecimationMatrix(N,M,L,Delta)
H = T2.dot(P.dot(T1))

data = data.transpose(2,1,0)
f_img = np.reshape(data,(N*M*L,))

# print(f_img)

# im_Dv = D.dot(im_v)
# im_D = np.reshape(im_Dv,(int(N/Delta),int((M+L-1)/Delta)))
# im_Ds = im_D#.swapaxes(0,1)

X = T2.dot(P.dot(T1))
# X = P
Af = lambda theta: A_forward(theta,X,N,M,L,Delta)
Ab = lambda g: A_backward(g,X,N,M,L,Delta)
# A = X
# A = linalg.LinearOperator(X.shape,matvec = Af,rmatvec = Ab)
A = lambda x,mode: Amatrix(x,mode,Af,Ab)
theta0 = dct3(f_img,N,M,L)
# print(np.mean(theta0))
# g = A.dot(theta0)
g = A(theta0,1)

# print((int(N/Delta),int((M+L-1)/Delta)))
# print(g.size)

im = np.reshape(g,(int((M+L-1)/Delta),int(N/Delta)))
im = im.swapaxes(0,1)
# im = np.reshape(g,((M+L-1), N))

# U = lambda x: dct3(x,N,M,L)
# Ut = lambda X: idct3(X,N,M,L)

# opts = {'U':sparse.identity(N*M*L),'normU':1}

# muf = 1e-8
# sigma = 0.001
# delta = np.sqrt(g.size + 2*np.sqrt(2*g.size))*sigma
# delta = 1e-8
# theta,niter,residuals,outputData = NESTA(A=Af,At=Ab,b=g,muf=muf,delta=delta,opts = opts)

# result = linalg.lsmr(A,g,damp=0.1,show=1)

tau = 0
theta,resid,grad,info = spg_lasso(A, g, tau)

# theta = result[0]

f_v = idct3(theta,N,M,L)
f = np.reshape(f_v,(L,N,M))  
f = f.transpose(1,2,0) 
data = data.transpose(2,1,0)

# img = im.mean(axis=2)

# print(f.mean(axis=2))

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()

ax1 = f1.add_subplot(111)
ax2 = f2.add_subplot(111)
ax3 = f3.add_subplot(111)

ax1.imshow(data.mean(axis=2),cmap='Greys_r',interpolation='none')
ax2.imshow(f.mean(axis=2),cmap='Greys_r',interpolation='none')
ax3.imshow(im,cmap='Greys_r',interpolation='none')

plt.show()

# print(np.sum((f - data)**2))
# print(f)

print(P.dot(f_img).reshape((M+L-1,N)).swapaxes(0,1))
print(f)
print(np.sum(theta0<0.1))