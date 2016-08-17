# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 11:15:36 2016
function test

@author: Chen
"""
import scipy.io as sio
from scipy.constants import pi
import numpy as np
import kernal_decoder as my_kd
import channel_package as my_cp
import channel_coding_package as my_ccp
import constellation_package as my_constell
#import pyldpc
#y=np.random.normal(0,2,(2,3))+1j*np.random.normal(0,2,(2,3))
#
##y=np.array([-1.61])
#ADC_2bit=my_cp.ADC_model(2,3)
#out_real,out_imag=ADC_2bit.quantization(y)
#real_upper,real_lower,imag_upper,imag_lower=ADC_2bit.find_bound(out_real,out_imag)

t=324
k=324
nt=2
nr=8
###
###
#mu_x=np.random.normal(0,1,(k,nt))+1j*np.random.normal(0,1,(k,nt))
#mu_x=np.ones((k,nt))*(1+1j)/np.sqrt(2)

##
#mu_h=np.random.normal(0,1,(k,nt,nr))
#delta_h=np.abs(np.random.normal(0,1,(k,nt,nr)))
##
##mu_y_b=np.random.normal(0,1,(t,k,nt,nr))
##delta_y_b=np.abs(np.random.normal(0,1,(t,k,nt,nr)))
##
#decoder1=my_kd.kernal_decoder(t,k,nt,nr,1,1)
##mu_b,delta_b=decoder1.update_xh_to_b(mu_x,delta_x,mu_h,delta_h)
##mu_b_y,delta_b_y=decoder1.update_b_to_y(mu_b,delta_b,mu_y_b,delta_y_b)
#y=np.random.normal(0,1,(t,nr))+1j*np.random.normal(0,1,(t,nr))
#ADC_2bit=my_cp.ADC_model(2,3)
#out_real,out_imag=ADC_2bit.quantization(y)
#digital_y=out_real+1j*out_imag
#real_upper,real_lower,imag_upper,imag_lower=ADC_2bit.find_bound(out_real,out_imag)
##y_real_upper=np.random.normal(0,1,(t,nr))
##y_real_lower=np.random.normal(0,1,(t,nr))
##y_imag_upper=np.random.normal(0,1,(t,nr))
##y_imag_lower=np.random.normal(0,1,(t,nr))
##mu_y_b,delta_y_b=decoder1.update_y_to_b(mu_b_y,delta_b_y,y,y_real_upper,y_real_lower,y_imag_upper,y_imag_lower)
##updated_mu_x,updated_delta_x,updated_mu_h,updated_delta_h=decoder1.update_b_to_xh(mu_y_b,delta_y_b,mu_x,delta_x,mu_h,delta_h)
#updated_mu_x,updated_delta_x,updated_mu_h,updated_delta_h=decoder1.integraded_decoder_no_loop(mu_x,delta_x,mu_h,delta_h,digital_y,real_upper,real_lower,imag_upper,imag_lower)

#import constellation_package as my_constell
#constell_qpsk=my_constell.constellation('default')
#bit_LLR=constell_qpsk.soft_demapping(mu_x,delta_x)
##interleaver1=my_constell.interleaver(k*2,10)
##a=interleaver1.interleave(bit_LLR)
##b=interleaver1.deinterleave(a)
#mu_sym,delta_sym=constell_qpsk.soft_mapping(bit_LLR)



H=sio.loadmat('ParityCheck.mat')['H']
G=sio.loadmat('Generator.mat')['G']

#H = pyldpc.RegularH(648,4,8)
#tG = pyldpc.CodingMatrix(H)
LDPC1=my_ccp.LDPC(H,G)
interleaver1=my_constell.interleaver(648,215)
qpsk=my_constell.constellation('default')
ADC_2bit=my_cp.ADC_model(0.5,4)
AMP_decoder=my_kd.kernal_decoder(t,k,nt,nr,0.05**2*2,1)

x_bit=np.random.randint(2, size=(324,nt))
temp1=np.zeros((648,nt))
temp2=np.zeros((648,nt))

for i in xrange(nt):
    en_x=LDPC1.encoder(x_bit[:,i]) 
    #en_x=pyldpc.Coding(tG,x_bit[:,i],1)
    en_in_x=interleaver1.interleave(en_x)
    bit_LLR=np.zeros(648)
    bit_LLR[en_in_x<0.5]=100
    bit_LLR[en_in_x>0.5]=-100
    temp1[:,i]=bit_LLR
original_bit=np.array(temp1<0).astype(int)    
x,delta_s=qpsk.soft_mapping(temp1)
dft=np.arange(t).reshape(t,1).dot(np.arange(k).reshape(1,k))
dft=np.exp(2*pi*1j*dft/k)
dft_extand=dft[:,:,None,None]+np.zeros((nt,nr))[None,None,:,:]
x_extand=x[None,:,:,None]+np.zeros((t,nr))[:,None,None,:]

channel=np.random.normal(0,1,(k,nt,nr))+1j*np.random.normal(0,1,(k,nt,nr))
channel_extand=channel[None,:,:,:]+np.zeros(t)[:,None,None,None]


noise=(np.random.normal(0,1,(t,nr))+1j*np.random.normal(0,1,(t,nr)))*0.05
y=np.sum(x_extand*channel_extand*dft_extand,axis=(1,2))/k+noise
quan_y_real,quan_y_imag=ADC_2bit.quantization(y)
quan_y=quan_y_real+1j*quan_y_imag
real_upper,real_lower,imag_upper,imag_lower=ADC_2bit.find_bound(quan_y_real,quan_y_imag)

p_noise=(np.random.normal(0,1,(t,nt,nr))+1j*np.random.normal(0,1,(t,nt,nr)))*0.01
py=np.sum(np.ones((t,k,nt,nr))*channel_extand*dft_extand,axis=1)/k+p_noise
quan_py_real,quan_py_imag=ADC_2bit.quantization(py)
quan_py=quan_py_real+1j*quan_py_imag
p_real_upper,p_real_lower,p_imag_upper,p_imag_lower=ADC_2bit.find_bound(quan_py_real,quan_py_imag)
mu_h_py=np.zeros((k,nt,nr))
delta_h_py=np.ones((k,nt,nr))

mu_h,delta_h=AMP_decoder.py_to_h(mu_h_py,delta_h_py,quan_py,p_real_upper,p_real_lower,p_imag_upper,p_imag_lower)

#mu_x=np.random.normal((k,nt))*1+1j*np.random.normal((k,nt))*1
mu_x=np.zeros((k,nt))
#mu_x=x
delta_x=np.ones((k,nt))*1
#delta_h=np.ones((k,nt,nr))*0.001


num_iter=5
ber=np.ones(num_iter)*648
for iteration in xrange(num_iter):
    updated_mu_x,updated_delta_x,updated_mu_h,updated_delta_h= \
    AMP_decoder.integraded_decoder_no_loop(mu_x,delta_x,mu_h,delta_h,quan_y,real_upper,real_lower,imag_upper,imag_lower)
    
    AMP_bit_LLR=qpsk.soft_demapping(updated_mu_x,updated_delta_x)
    
    
    for i in xrange(nt):
        LDPC_in=interleaver1.deinterleave(AMP_bit_LLR[:,i])
        LDPC_out=LDPC1.decoder(LDPC_in,10)
        #LDPC_out=pyldpc.Decoding_logBP(H,LDPC_in,1,10)
        new_AMP_LLR=interleaver1.interleave(LDPC_out)    
        temp2[:,i]=new_AMP_LLR
    decoded_bit=np.array(temp2<0).astype(int)
    ber[iteration]=np.sum((decoded_bit+original_bit)%2,axis=(0,1))
    mu_x,delta_x=qpsk.soft_mapping(temp2)




"""
"""
#H = pyldpc.RegularH(15,4,5)
#tG = pyldpc.CodingMatrix(H)
#LDPC1=my_ccp.LDPC(H,G)
#mu_s=(np.random.normal(0,0.4,(64,1))+1j*np.random.normal(0,0.4,(64,1)))
#new_bit_LLR=qpsk.soft_demapping(mu_s,np.ones((64,1))*0.8)
#d_LLR=interleaver1.deinterleave(new_bit_LLR[:,0])

#v = np.random.randint(2,size=324)
#y = pyldpc.Coding(G.T,v,1).astype(float)
#s=y.copy()
#y[y==1]=-10.0
#y[y==0]=10.0
#
#y=np.random.normal(0,10,(648))
###
##h_nd_x=np.array(d_LLR<= 0).astype(int)
#d_x1=pyldpc.Decoding_logBP(H,y,1,5)
#d_x=LDPC1.decoder(d_LLR,10)
#
#h_x=np.array(d_x <= 0).astype(int)
#e=np.sum((h_x[0:324]+x)%2)
#en=np.sum((h_nd_x[0:324]+x)%2)
#a=en_x+np.random.normal(0,0.001,648)
##x_decoded = pyldpc.Decoding_logBP(H,a,-10*np.log10(2),5)
#y_decoded = LDPC1.decoder(a,5)

