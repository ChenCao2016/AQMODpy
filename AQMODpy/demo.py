# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:08:29 2016

This is a simple demo for XXX

@author: Chen Cao, University of Lousiville

"""
import sys
def printProgress (BER,iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s, BER in %d iteration is %.4f' % (prefix, bar, percents, '%', suffix, iteration,BER)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
###############################################################################




import scipy.io as sio
from scipy.constants import pi
import numpy as np
import kernal_decoder as my_kd
import channel_package as my_cp
import channel_coding_package as my_ccp
import constellation_package as my_constell

###############################################################################
"""
t: number of snapshots
k: number of subcarriers
nt: number of transmitters (users)
nr: number of receiver antennas
"""
t=324
k=324
nt=10
nr=30

SNR=25.0
###############################################################################

"""
H: Paritycheck matrix
G: Generator matrix

IEEE 802.11, 1/2 rate, (648,324) LDPC codes
"""
H=sio.loadmat('ParityCheck.mat')['H']
G=sio.loadmat('Generator.mat')['G']
LDPC1=my_ccp.LDPC(H,G)
###############################################################################


"""
random block interleaver, size 648
"""
interleaver1=my_constell.interleaver(648,215)
###############################################################################

"""
constellation QPSK
"""
qpsk=my_constell.constellation('default')
###############################################################################

"""
symmetric ADC model
scale: work as auto gain controller (AGC) amplifier to fit the input scale of ADC
resolution: number of ADC bits per sample
"""
AGC_scale=0.5
resolution=4
ADC1=my_cp.ADC_model(AGC_scale,resolution)
###############################################################################

"""
AMP MIMO OFDM decoder
"""
AMP_decoder=my_kd.kernal_decoder(t,k,nt,nr,1/(10**(SNR/10.0)),1)
###############################################################################


"""
Transmit and channel process
"""
print 'Encoding starts'

x_bit=np.random.randint(2, size=(324,nt)) # source bit
temp1=np.zeros((648,nt))


# encoding and interleaving
for i in xrange(nt):  
    en_x=LDPC1.encoder(x_bit[:,i]) 
    en_in_x=interleaver1.interleave(en_x)
    bit_LLR=np.zeros(648)
    bit_LLR[en_in_x<0.5]=100
    bit_LLR[en_in_x>0.5]=-100
    temp1[:,i]=bit_LLR
original_bit=np.array(temp1<0).astype(int) 


#constellation mapping
x,delta_s=qpsk.soft_mapping(temp1) 


#IDFT matrix for OFDM 
dft=np.arange(t).reshape(t,1).dot(np.arange(k).reshape(1,k))
dft=np.exp(2*pi*1j*dft/k)
dft_extand=dft[:,:,None,None]+np.zeros((nt,nr))[None,None,:,:]


#OFDM channel realization, you can change to use other models
x_extand=x[None,:,:,None]+np.zeros((t,nr))[:,None,None,:]
channel=np.random.normal(0,1,(k,nt,nr))+1j*np.random.normal(0,1,(k,nt,nr))
channel_extand=channel[None,:,:,:]+np.zeros(t)[:,None,None,None]
noise=(np.random.normal(0,1,(t,nr))+1j*np.random.normal(0,1,(t,nr)))*np.sqrt(1/(10**(SNR/10.0))/2)
y=np.sum(x_extand*channel_extand*dft_extand,axis=(1,2))/k+noise


#ADC quantization
quan_y_real,quan_y_imag=ADC1.quantization(y)
quan_y=quan_y_real+1j*quan_y_imag
real_upper,real_lower,imag_upper,imag_lower=ADC1.find_bound(quan_y_real,quan_y_imag)



#pilot referance signals transmission, (only one slot adpot in this demo)
p_noise=(np.random.normal(0,1,(t,nt,nr))+1j*np.random.normal(0,1,(t,nt,nr)))*np.sqrt(1/(10**(SNR/10.0))/2)
py=np.sum(np.ones((t,k,nt,nr))*channel_extand*dft_extand,axis=1)/k+p_noise


# ADC quantization
quan_py_real,quan_py_imag=ADC1.quantization(py)
quan_py=quan_py_real+1j*quan_py_imag
p_real_upper,p_real_lower,p_imag_upper,p_imag_lower=ADC1.find_bound(quan_py_real,quan_py_imag)

###############################################################################
print 'Done'

"""
Decoding starts
"""
print 'Decoding starts \n'

# estimate OFDM channel
mu_h_py=np.zeros((k,nt,nr))
delta_h_py=np.ones((k,nt,nr))
mu_h,delta_h=AMP_decoder.py_to_h(mu_h_py,delta_h_py,quan_py,p_real_upper,p_real_lower,p_imag_upper,p_imag_lower)


# data symbols decoding, only one OFDM symbol in this demo
num_iter=7 #number of iteration
ber=np.ones(num_iter) # bit error rate indicator
mu_x=np.zeros((k,nt))
delta_x=np.ones((k,nt))
temp2=np.zeros((648,nt))
for iteration in xrange(num_iter):
    printProgress(ber[iteration-1 if iteration>0 else 0],iteration, num_iter, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
    updated_mu_x,updated_delta_x,updated_mu_h,updated_delta_h= \
    AMP_decoder.integraded_decoder_no_loop(mu_x,delta_x,mu_h,delta_h,quan_y,real_upper,real_lower,imag_upper,imag_lower)
    AMP_bit_LLR=qpsk.soft_demapping(updated_mu_x,updated_delta_x)   
    for i in xrange(nt):
        LDPC_in=interleaver1.deinterleave(AMP_bit_LLR[:,i])
        LDPC_out=LDPC1.decoder(LDPC_in,10)
        new_AMP_LLR=interleaver1.interleave(LDPC_out)    
        temp2[:,i]=new_AMP_LLR
    decoded_bit=np.array(temp2<0).astype(int)
    ber[iteration]=np.sum((decoded_bit+original_bit)%2,axis=(0,1))/(648.0*nt)
    if ber[iteration]<(10**-30):
        break
    mu_x,delta_x=qpsk.soft_mapping(temp2)
    

printProgress(ber[iteration],iteration+1, iteration+1, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
#final result, BER performance
print '\nBER is '+np.array_str(ber[0:(iteration+1)])


