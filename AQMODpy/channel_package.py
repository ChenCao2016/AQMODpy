# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 13:23:56 2016

Channel package

including: symmetric ADC model

@author: Chen Cao, University of Louisville
"""


class ADC_model(object):
    # assume the ADC mapping is symmetric. You can use different ADC models
    def __init__(self,scale,resolution):
        """
        scale: work as auto gain controller (AGC) amplifier to fit the input scale of ADC
        resolution: number of ADC bits per sample
        """
        self.num_steps=2**resolution-1 
        self.len_step=(2.0*scale)/(self.num_steps-2)
        self.half_len_step=self.len_step/2.0
        self.upper=(self.num_steps-1)/2*self.len_step
        self.lower=-self.upper
        
    def quantization(self,analog_input):
        """
        quantizaion process
        
        input: 
        analog_input: unquantized signal, any dimension
        
        output:
        out_real: quantized real part, dimension corresponding to input
        out_imag: quantized imag part, dimension corresponding to input
        """        
        
        input_real=analog_input.real
        input_imag=analog_input.imag  
        
        temp=(input_real/self.half_len_step)
        temp=temp.astype(int)
        temp[temp>0]+=1
        #temp[temp<0]-=1 #take notice of integer operation
        out_real=temp/2*self.len_step
        out_real[out_real>self.upper]=self.upper
        out_real[out_real<self.lower]=self.lower
        
        temp=(input_imag/self.half_len_step)
        temp=temp.astype(int)
        temp[temp>0]+=1
        #temp[temp<0]-=1 #take notice of integer operation
        out_imag=temp/2*self.len_step
        out_imag[out_imag>self.upper]=self.upper
        out_imag[out_imag<self.lower]=self.lower
        
        return out_real,out_imag
        
    def find_bound(self,digital_real,digital_imag):
        """
        find the unquantized bound corresponding to the quantized signal
        
        input:
        digital_real: quantized real part, any dimension
        digital_imag: quantized imag part, the same dimension as digital_real
        
        
        output:
        real_upper,real_lower,imag_upper,imag_lower: corresponding bounds
        """        
        
        #use arbitrary large number: 100
        real_upper=digital_real+self.half_len_step
        real_upper[real_upper>self.upper]=100.0
        real_lower=digital_real-self.half_len_step
        real_lower[real_lower<self.lower]=-100.0
        imag_upper=digital_imag+self.half_len_step
        imag_upper[imag_upper>self.upper]=100.0
        imag_lower=digital_imag-self.half_len_step
        imag_lower[imag_lower<self.lower]=-100.0
        
        return real_upper,real_lower,imag_upper,imag_lower