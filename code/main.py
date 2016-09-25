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
import spectral
import csv
from scipy.interpolate import interp1d

# All based on work from http://www.scielo.org.co/pdf/iei/v34n3/v34n3a09.pdf

N0 = 1160
M0 = 320
L0 = 360

zfactor = 0.5
Nmax = 1024
Mmax = 256
Lmax = int((128*zfactor+1)/zfactor)

N = int(Nmax*zfactor)
M = int(Mmax*zfactor)
L = int(Lmax*zfactor)

Delta = 1

np.random.seed(0)

imdir = "../images/DeepHorizon_OilSpill/"
imname = "0612-1615_rad_sub.dat"
impath = imdir + imname

def readData(fname,N0,M0,L0,zfactor,Nmax,Mmax,Lmax):
	data = np.fromfile(fname,dtype=np.int16)
	data = np.reshape(data,(N0,L0,M0))
	data = np.swapaxes(data,1,2)
	data = zoom(data[0:Nmax,0:Mmax,0:Lmax],zfactor)
	return data

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

# def genDecimationMatrix(N,M,L,Delta):
# 	data0 = np.ones((Delta**2,))
# 	r0 = np.zeros((Delta**2,))
# 	c0 = np.arange(Delta**2)
# 	data = np.ones((N*(M+L-1),))
# 	r = np.empty((N*(M+L-1),))
# 	c = np.empty((N*(M+L-1),))
# 	for i in range(int((M+L-1)/Delta)):
# 		for j in range(int(N/Delta)):
# 			r[i*N*Delta + j*Delta**2:i*N*Delta + j*Delta**2 + Delta**2] = r0 + i*N/Delta + j
# 			c[i*N*Delta + j*Delta**2:i*N*Delta + j*Delta**2 + Delta**2] = c0 + j*Delta + i*Delta**2
# 	D = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1)/Delta**2,N*(M+L-1)))
# 	return D

def genDecimationMatrix(N,M,L,Delta):
	data = 1./Delta**2 * np.ones(((N*(M+L-1),)))
	c0 = np.arange(Delta)
	r = np.empty((N*(M+L-1),))
	c = np.empty((N*(M+L-1),))
	nn = 0
	for i in range(int((M+L-1)/Delta)):
		for j in range(int(N/Delta)):
			r[nn:nn+Delta**2] = np.tile((i*int(N/Delta) + j)*np.ones((Delta,)),Delta)
			# print('r')
			# print((i*int(N/Delta) + j)*np.ones((Delta,)))
			cadd = np.empty((Delta**2,))
			for d in range(Delta):
				cadd[d*Delta:(d+1)*Delta] = (i*N + j)*Delta + d*N + c0
				# print('c')
				# print((i*N + j)*Delta + d*N + c0)
			c[nn:nn + Delta**2] = cadd
			nn += Delta**2
	# print(c)
	D = sparse.csr_matrix((data,(r,c)),shape=(N*(M+L-1)/Delta**2,N*(M+L-1)))
	return D


def dct3(x,N,M,L):
	x = np.reshape(x,(N,M,L))
	X = dct(dct(dct(x,norm='ortho').transpose(0,2,1),norm='ortho').transpose(1,2,0),norm='ortho').transpose(2,0,1).transpose(0,2,1)
	X = np.reshape(X,(N*M*L,))
	return X

def idct3(X,N,M,L):
	X = np.reshape(X,(N,M,L))
	x = idct(idct(idct(X,norm='ortho').transpose(0,2,1),norm='ortho').transpose(1,2,0),norm='ortho').transpose(2,0,1).transpose(0,2,1)
	x = np.reshape(x,(N*M*L,))
	return x

def A_forward(theta,X,N,M,L,Delta):
	f = idct3(theta,N,M,L)
	g = X.dot(f)
	# g_im = g.reshape((M+L-1,N))
	# plt.imshow(g_im,cmap='Greys_r',interpolation='none')
	# plt.show()
	# g_d = imresize(g_im,1./Delta)
	# g_v = g_d.reshape((int(N*(M+L-1)/Delta**2),))
	return g

def A_backward(g,X,N,M,L,Delta):
	# g_d = g.reshape((int((M+L-1)/Delta),int(N/Delta)))
	# plt.imshow(g_d,cmap='Greys_r',interpolation='none')
	# plt.show()
	# g_im = zoom(g_d,Delta,order=0)
	# g = g_im.reshape((N*(M+L-1),))
	Xt = X.T
	f = Xt.dot(g)
	theta = dct3(f,N,M,L)
	return theta

def Amatrix(x,mode,Af,Ab):
	if mode == 1:
		return Af(x)
	else:
		return Ab(x)

def importAndInterpolateResponse(fnameResponse,fnameCubeWavelenths):
    response_data = np.genfromtxt(fnameResponse, delimiter=',')
    response_data = np.nan_to_num(response_data)
    cube_wavelengths = np.genfromtxt(fnameCubeWavelenths, delimiter=',')
    lam = response_data[:,0]
    maxlam = np.max(lam)
    res = response_data[:,1:4]
    r_res = res[:,0]
    g_res = res[:,1]
    b_res = res[:,2]
    fr = interp1d(lam,r_res)
    fg = interp1d(lam,g_res)
    fb = interp1d(lam,b_res)
    cw_trunc = cube_wavelengths[np.where(cube_wavelengths<maxlam)]
    r_new = np.zeros(cube_wavelengths.shape)
    g_new = np.zeros(cube_wavelengths.shape)
    b_new = np.zeros(cube_wavelengths.shape)
    r_new[0:len(cw_trunc)] = fr(cw_trunc)
    g_new[0:len(cw_trunc)] = fg(cw_trunc)
    b_new[0:len(cw_trunc)] = fb(cw_trunc)
    newres = np.stack((r_new.T,g_new.T,b_new.T),1)
    return newres
    

def hypercube2rgb(cube,res):
    r_res = res[:,0]
    g_res = res[:,1]
    b_res = res[:,2]
    r_res /= np.sum(r_res)
    g_res /= np.sum(g_res)
    b_res /= np.sum(b_res)
    R = np.zeros((cube.shape[0],cube.shape[1]))
    G = np.zeros((cube.shape[0],cube.shape[1]))
    B = np.zeros((cube.shape[0],cube.shape[1]))
    for b in range(cube.shape[2]):
        for i in range(cube.shape[0]):
            for j in range(cube.shape[1]):
                R[i,j] += r_res[b] * cube[i,j,b]
                G[i,j] += g_res[b] * cube[i,j,b]
                B[i,j] += b_res[b] * cube[i,j,b]
    rgb_im = np.stack((R,G,B),2)
    rgb_im /= np.max(rgb_im)
    rgb_im *= 256
    rgb_im = np.uint8(rgb_im)
    return rgb_im
                

res = importAndInterpolateResponse('linss2_10e_fine.csv','cube_wavelengths.csv')
data = readData(impath,N0,M0,L0,zfactor,Nmax,Mmax,Lmax)
rgb_im = hypercube2rgb(data,res)

fim = plt.figure()
axim = fim.add_subplot(111)
axim.imshow(rgb_im,interpolation='none')

#
## N = 16
## M = 16
## L = 9
## Delta = 4
#
## # l1 = np.ones((N,M))
## # data = np.zeros((N,M,L))
## data = np.ones((N,M,L))
## data[5:10,3:8,:] = np.zeros((5,5,9))
## im_data = np.random.rand(N,M+L-1)
## im_v =im_data.reshape((N*(M+L-1),))
## data = np.stack((l1,2*l1,3*l1,l1,0.5*l1,l1,0.5*l1),axis=2)
#
#T1 = genMask1Matrix(N,M,L)
#T2 = genMask2Matrix(N,M,L)
#P = genDispMatrix(N,M,L)
#D = genDecimationMatrix(N,M,L,Delta)
#H = T2.dot(P.dot(T1))
#
#data = data.transpose(0,2,1)
#f_img = np.reshape(data,(N*M*L,))
#
## print(f_img)
#
## im_Dv = D.dot(im_v)
## im_D = np.reshape(im_Dv,(int(N/Delta),int((M+L-1)/Delta)))
## im_Ds = im_D#.swapaxes(0,1)
#
## X = D.dot(T2.dot(P.dot(T1)))
#X = P.dot(T1)
## X = D.dot(P)
## X = P
#Af = lambda theta: A_forward(theta,X,N,M,L,Delta)
#Ab = lambda g: A_backward(g,X,N,M,L,Delta)
## A = X
## A = linalg.LinearOperator(X.shape,matvec = Af,rmatvec = Ab)
#A = lambda x,mode: Amatrix(x,mode,Af,Ab)
#theta0 = dct3(f_img,N,M,L)
## print(np.mean(theta0))
## g = A.dot(theta0)
#g = A(theta0,1)
#
## print((int(N/Delta),int((M+L-1)/Delta)))
## print(g.size)
#
#im = np.reshape(g,(int((M+L-1)/Delta),int(N/Delta)))
#im = im.swapaxes(0,1)
## im = np.reshape(g,((M+L-1), N))
#
## U = lambda x: dct3(x,N,M,L)
## Ut = lambda X: idct3(X,N,M,L)
#
## opts = {'U':sparse.identity(N*M*L),'normU':1}
#
## muf = 1e-8
## sigma = 0.001
## delta = np.sqrt(g.size + 2*np.sqrt(2*g.size))*sigma
## delta = 1e-8
## theta,niter,residuals,outputData = NESTA(A=Af,At=Ab,b=g,muf=muf,delta=delta,opts = opts)
#
## result = linalg.lsmr(A,g,damp=0.1,show=1)
#opts = spgSetParms({'verbosity':2})
#tau = 0
#theta,resid,grad,info = spg_lasso(A, g, tau,opts)
## print(info)
#
## theta = result[0]
#
#f_v = idct3(theta,N,M,L)
#f = np.reshape(f_v,(N,L,M))  
#f = f.transpose(0,2,1) 
#data = data.transpose(0,2,1)
#
## img = im.mean(axis=2)
#
## print(f.mean(axis=2))
#
#f1 = plt.figure()
## f2 = plt.figure()
#f3 = plt.figure()
#f4 = plt.figure()
#f5 = plt.figure()
#
#ax1 = f1.add_subplot(111)
## ax2 = f2.add_subplot(111)
#ax3 = f3.add_subplot(111)
#ax4 = f4.add_subplot(111)
#ax5 = f5.add_subplot(111)
#
#ax1.imshow(data.mean(axis=2),cmap='Greys_r',interpolation='none',vmin=0,vmax=2048)
## ax2.imshow(T2.dot(P.dot(T1.dot(f_img))).reshape(((M+L-1), N)).swapaxes(1,0),cmap='Greys_r',interpolation='none')
#ax3.imshow(im,cmap='Greys_r',interpolation='none')
#ax4.imshow(D.todense(),cmap='Greys_r',interpolation='none')
#ax5.imshow(f.mean(axis=2),cmap='Greys_r',interpolation='none',vmin=0,vmax=2048)
#
#
#plt.show()
#spectral.imshow(data,(1,2,3))
#
#print(np.sum((f_img - f_v)**2))
#print(f_img)
## print(data)
#print(f)
#print(P.dot(f_img).reshape((M+L-1,N)))
#print(D.dot(P.dot(f_img)))
#print(im)
## print(idct(dct(f_img,norm='ortho'),norm='ortho'))
## print(np.sum(theta0<0.1))