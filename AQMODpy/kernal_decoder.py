# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 09:57:51 2016

kernal decoder (GAMP module)


You can use integraded_decoder(self,mu_x,delta_x,mu_h,delta_h,y,y_real_upper,y_real_lower,y_imag_upper,y_imag_lower),
or integraded_decoder_no_loop(self,mu_x,delta_x,mu_h,delta_h,y,y_real_upper,y_real_lower,y_imag_upper,y_imag_lower)
for demonstrating entire GAMP decoder.

Or you can check each message passing steps use individual sub functions


For corresponding channel estimation, plz use function:
py_to_h(self,mu_h_py,delta_h_py,py,py_real_upper,py_real_lower,py_imag_upper,py_imag_lower)



t: number of snapshots
k: number of subcarriers
nt: number of transmitter antennas (users)
nr: number of receiver antennas

@author: Chen Cao, University of Louisville

"""
import numpy as np
from scipy.stats import norm
from scipy.constants import pi

class kernal_decoder(object):
    
    def __init__(self,t,k,nt,nr,var_n,var_b):
        self.t=t;self.k=k;self.nt=nt;self.nr=nr
        self.var_n=var_n;self.var_b=var_b
        self.IDFT=np.arange(t).reshape(t,1).dot(np.arange(k).reshape(1,k))
        self.IDFT=np.exp(2*pi*1j*self.IDFT/k)
        self.IDFT_extand=self.IDFT[:,:,None,None]+np.zeros((nt,nr))[None,None,:,:]
        
        #This is only required for decoder with loop
        #These two variable matrix are attributes of object, because the decoder with loop requires the memory 
        self.mu_y_b=np.zeros((t,k,nt,nr))
        self.delta_y_b=np.ones((t,k,nt,nr))*10
        
        
        
    def update_xh_to_b(self,mu_x,delta_x,mu_h,delta_h):
        """
        input:
        mu_x: expectation, dimension k*nt
        delta_x: variance, dimension k*nt
        mu_h: expectation, dimension k*nt*nr
        delta_h: variance, dimension k*nt*nr
        
        output:
        mu_b: expectation, dimension k*nt*nr
        delta_b:expectation, dimension k*nt*nr
        
        """

        
        mu_b=(mu_x[...,None]+np.zeros(self.nr)[None,None,:])*mu_h
        delta_b=(delta_x[...,None]+np.zeros(self.nr)[None,None,:])*delta_h \
                +((np.abs(mu_x)**2)[...,None]+np.zeros(self.nr)[None,None,:])*delta_h \
                +(np.abs(mu_h)**2)*(delta_x[...,None]+np.zeros(self.nr)[None,None,:])   

        return mu_b,delta_b
        
    def update_b_to_y(self,mu_b,delta_b,mu_y_b,delta_y_b):
        """
        input:
        mu_b: expectation, dimension k*nt*nr
        delta_b: variance, dimension k*nt*nr
        mu_y_b: expectation, dimension t*k*nt*nr
        delta_y_b: variance, dimension t*k*nt*nr
        
        output:
        mu_b_y: expectation, dimension t*k*nt*nr
        delta_b_y: variance, dimension t*k*nt*nr
        """

        temp_sum_delta=np.sum(1/delta_y_b,axis=0)+1/delta_b
        temp_sum_mu=np.sum(mu_y_b/delta_y_b,axis=0)+mu_b/delta_b
        
        delta_b_y=1/(temp_sum_delta[None,...]+np.zeros((self.t))[:,None,None,None]-1/delta_y_b)
        mu_b_y=(temp_sum_mu[None,...]+np.zeros((self.t))[:,None,None,None]-mu_y_b/delta_y_b)*delta_b_y
            
        return mu_b_y,delta_b_y
        
    def update_b_to_y_no_loop(self,mu_b,delta_b):
        """
        input:
        mu_b: expectation, dimension k*nt*nr
        delta_b: variance, dimension k*nt*nr
        mu_y_b: expectation, dimension t*k*nt*nr
        delta_y_b: variance, dimension t*k*nt*nr
        
        output:
        mu_b_y: expectation, dimension t*k*nt*nr
        delta_b_y: variance, dimension t*k*nt*nr
        """

        mu_b_y=mu_b[None,...]+np.zeros((self.t))[:,None,None,None]
        delta_b_y=delta_b[None,...]+np.zeros((self.t))[:,None,None,None]
            
        return mu_b_y,delta_b_y
    
    def update_y_to_b(self,mu_b_y,delta_b_y,y,y_real_upper,y_real_lower,y_imag_upper,y_imag_lower):
        """
        input:
        mu_b_y: expectation, dimension t*k*nt*nr
        delta_b_y: expectation, dimension t*k*nt*nr
        y: quantized digital signals, dimension t*nr 
        
        output:
        mu_y_b: expectation, dimension t*k*nt*nr
        delta_y_b: variance, dimension t*k*nt*nr        
        """

        DFT_mu_b_y=mu_b_y*self.IDFT_extand

        temp_sum_mu=np.sum(DFT_mu_b_y,axis=(1,2))[:,None,None,:]+np.zeros((self.k,self.nt))[None,:,:,None]
        temp_sum_delta=np.sum(delta_b_y,axis=(1,2))[:,None,None,:]+np.zeros((self.k,self.nt))[None,:,:,None]
        
        temp_sum_slash_mu=(temp_sum_mu-DFT_mu_b_y)/self.k
        temp_sum_slash_mu_real=temp_sum_slash_mu.real
        temp_sum_slash_mu_imag=temp_sum_slash_mu.imag
        temp_sum_slash_delta=(temp_sum_delta-delta_b_y)/self.k**2
        
        y_real_upper_extand=y_real_upper[:,None,None,:]+np.zeros((self.k,self.nt))[None,:,:,None] 
        y_real_lower_extand=y_real_lower[:,None,None,:]+np.zeros((self.k,self.nt))[None,:,:,None] 
        y_imag_upper_extand=y_imag_upper[:,None,None,:]+np.zeros((self.k,self.nt))[None,:,:,None] 
        y_imag_lower_extand=y_imag_lower[:,None,None,:]+np.zeros((self.k,self.nt))[None,:,:,None]
        
        y_extand=y[:,None,None,:]+np.zeros((self.k,self.nt))[None,:,:,None] 
        
        gamma_real_nominator=norm.pdf((y_real_lower_extand-temp_sum_slash_mu_real)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b)) \
                             -norm.pdf((y_real_upper_extand-temp_sum_slash_mu_real)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b))       
        
#        gamma_real_nominator=norm.pdf(y_real_lower_extand,loc=temp_sum_slash_mu_real,scale=temp_sum_slash_delta+self.var_n+self.var_b) \
#                             -norm.pdf(y_real_upper_extand,loc=temp_sum_slash_mu_real,scale=temp_sum_slash_delta+self.var_n+self.var_b)
        gamma_real_denominator=norm.sf((y_real_lower_extand-temp_sum_slash_mu_real)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b)) \
                             -norm.sf((y_real_upper_extand-temp_sum_slash_mu_real)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b))    

        gamma_imag_nominator=norm.pdf((y_imag_lower_extand-temp_sum_slash_mu_imag)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b)) \
                             -norm.pdf(( y_imag_upper_extand-temp_sum_slash_mu_imag)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b))  
                             
#        gamma_imag_nominator=norm.pdf(y_imag_lower_extand,loc=temp_sum_slash_mu_imag,scale=temp_sum_slash_delta+self.var_n+self.var_b) \
#                             -norm.pdf( y_imag_upper_extand,loc=temp_sum_slash_mu_imag,scale=temp_sum_slash_delta+self.var_n+self.var_b)
        gamma_imag_denominator=norm.sf((y_imag_lower_extand-temp_sum_slash_mu_imag)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b)) \
                             -norm.sf(( y_imag_upper_extand-temp_sum_slash_mu_imag)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b))
                             
        temp_mu_y_b=self.var_b*gamma_real_nominator/gamma_real_denominator+1j*self.var_b*gamma_imag_nominator/gamma_imag_denominator
        bar_mu_y_b=self.var_b*(y_extand-temp_sum_slash_mu)/(temp_sum_slash_delta+self.var_b+self.var_n)
        
        self.delta_y_b=self.k**2*(np.abs(temp_mu_y_b-bar_mu_y_b)**2+(temp_sum_slash_delta*self.var_b+self.var_b*self.var_n)/(temp_sum_slash_delta+self.var_b+self.var_n))
        self.mu_y_b=self.k*temp_mu_y_b*(self.IDFT_extand.conj())
        return self.mu_y_b,self.delta_y_b
    
    def update_b_to_xh(self,mu_y_b,delta_y_b,mu_x,delta_x,mu_h,delta_h):
        """
        input:
        mu_y_b: expectation, dimension t*k*nt*nr
        delta_y_b: variance, dimension t*k*nt*nr
        mu_x: expectation, dimension k*nt
        delta_x: variance, dimension k*nt
        mu_h: expectation, dimension k*nt*nr
        delta_h: variance, dimension k*nt*nr

        intermedia:
        mu_xh: expectation, dimension k*nt*nr
        delta_xh: variance, dimension k*nt*nr 
        
        output:
        updated_mu_x: expectation, dimension k*nt
        updated_delta_x: variance, dimension k*nt
        updated_mu_h: expectation, dimension k*nt*nr
        updated_delta_h: variance, dimension k*nt*nr        
        """
        delta_xh=1/np.sum(1/delta_y_b,axis=0)
        mu_xh=delta_xh*np.sum(mu_y_b/delta_y_b,axis=0)
        
        mu_x_unscale=mu_h.conj()*mu_xh/(delta_h+np.abs(mu_h)**2)
        delta_x_unscale=delta_xh/(delta_h+np.abs(mu_h)**2)
        
        updated_delta_x=1/np.sum(1/delta_x_unscale,axis=2)
        updated_mu_x=updated_delta_x*np.sum(mu_x_unscale/delta_x_unscale,axis=2)
        
        mu_x_extand=mu_x[...,None]+np.zeros((self.nr))[None,None,:]
        delta_x_extand=delta_x[...,None]+np.zeros((self.nr))[None,None,:]
        updated_mu_h=mu_x_extand.conj()*mu_xh/(delta_x_extand+np.abs(mu_x_extand)**2)
        updated_delta_h=delta_xh/(delta_x_extand+np.abs(mu_x_extand)**2)
        
        return updated_mu_x,updated_delta_x,updated_mu_h,updated_delta_h
        
        
    def py_to_h(self,mu_h_py,delta_h_py,py,py_real_upper,py_real_lower,py_imag_upper,py_imag_lower):
        """
        use quantized pilot reference signals for channel estimation
        
        input: 
        py: quantized digital signals, dimension t*nt*nr 
        mu_h_py: expectation, dimension k*nt*nr
        delta_h_py: variance, dimension k*nt*nr
        
        output:
        mu_h: expectation, dimension k*nt*nr
        delta_h: variance, dimension k*nt*nr        
        """
        DFT_mu_h_py=(mu_h_py[None,:,:,:]+np.zeros((self.t))[None,:,None,None])*self.IDFT_extand
        delta_h_py_extand=mu_h_py[None,:,:,:]+np.zeros((self.t))[None,:,None,None]
        temp_sum_mu=np.sum(DFT_mu_h_py,axis=1)[:,None,:,:]+np.zeros((self.k))[None,:,None,None]
        temp_sum_delta=np.sum(delta_h_py_extand,axis=0)[:,None,:,:]+np.zeros((self.k))[None,:,None,None]
        
        temp_sum_slash_mu=(temp_sum_mu-DFT_mu_h_py)/self.k
        temp_sum_slash_mu_real=temp_sum_slash_mu.real
        temp_sum_slash_mu_imag=temp_sum_slash_mu.imag
        temp_sum_slash_delta=(temp_sum_delta-delta_h_py_extand)/self.k**2
        
        py_real_upper_extand=py_real_upper[:,None,:,:]+np.zeros((self.k))[None,:,None,None] 
        py_real_lower_extand=py_real_lower[:,None,:,:]+np.zeros((self.k))[None,:,None,None] 
        py_imag_upper_extand=py_imag_upper[:,None,:,:]+np.zeros((self.k))[None,:,None,None] 
        py_imag_lower_extand=py_imag_lower[:,None,:,:]+np.zeros((self.k))[None,:,None,None]
        
        py_extand=py[:,None,:,:]+np.zeros((self.k))[None,:,None,None] 
        
        gamma_real_nominator=norm.pdf((py_real_lower_extand-temp_sum_slash_mu_real)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b)) \
                             -norm.pdf((py_real_upper_extand-temp_sum_slash_mu_real)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b))       
        
        gamma_real_denominator=norm.sf((py_real_lower_extand-temp_sum_slash_mu_real)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b)) \
                             -norm.sf((py_real_upper_extand-temp_sum_slash_mu_real)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b))    

        gamma_imag_nominator=norm.pdf((py_imag_lower_extand-temp_sum_slash_mu_imag)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b)) \
                             -norm.pdf((py_imag_upper_extand-temp_sum_slash_mu_imag)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b))  
                             
        gamma_imag_denominator=norm.sf((py_imag_lower_extand-temp_sum_slash_mu_imag)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b)) \
                             -norm.sf((py_imag_upper_extand-temp_sum_slash_mu_imag)/np.sqrt(temp_sum_slash_delta+self.var_n+self.var_b))
                             
        temp_mu_py_h=self.var_b*gamma_real_nominator/gamma_real_denominator+1j*self.var_b*gamma_imag_nominator/gamma_imag_denominator
        bar_mu_py_h=self.var_b*(py_extand-temp_sum_slash_mu)/(temp_sum_slash_delta+self.var_b+self.var_n)
        
        delta_py_h=self.k**2*(np.abs(temp_mu_py_h-bar_mu_py_h)**2+(temp_sum_slash_delta*self.var_b+self.var_b*self.var_n)/(temp_sum_slash_delta+self.var_b+self.var_n))
        mu_py_h=self.k*temp_mu_py_h*(self.IDFT_extand.conj())
        
        delta_h=1/np.sum(1/delta_py_h,axis=0)
        mu_h=delta_h*np.sum(mu_py_h/delta_py_h,axis=0)      
             
        return mu_h,delta_h
        
        
        
    def integraded_decoder(self,mu_x,delta_x,mu_h,delta_h,y,y_real_upper,y_real_lower,y_imag_upper,y_imag_lower):
        mu_b,delta_b=self.update_xh_to_b(mu_x,delta_x,mu_h,delta_h)
        mu_b_y,delta_b_y=self.update_b_to_y(mu_b,delta_b,self.mu_y_b,self.delta_y_b)
        mu_y_b,delta_y_b=self.update_y_to_b(mu_b_y,delta_b_y,y,y_real_upper,y_real_lower,y_imag_upper,y_imag_lower)
        return self.update_b_to_xh(mu_y_b,delta_y_b,mu_x,delta_x,mu_h,delta_h)
        
    def integraded_decoder_no_loop(self,mu_x,delta_x,mu_h,delta_h,y,y_real_upper,y_real_lower,y_imag_upper,y_imag_lower):
        mu_b,delta_b=self.update_xh_to_b(mu_x,delta_x,mu_h,delta_h)
        mu_b_y,delta_b_y=self.update_b_to_y_no_loop(mu_b,delta_b)
        mu_y_b,delta_y_b=self.update_y_to_b(mu_b_y,delta_b_y,y,y_real_upper,y_real_lower,y_imag_upper,y_imag_lower)
        updated_mu_x,updated_delta_x,updated_mu_h,updated_delta_h=self.update_b_to_xh(mu_y_b,delta_y_b,mu_x,delta_x,mu_h,delta_h)
        return updated_mu_x,updated_delta_x,updated_mu_h,updated_delta_h