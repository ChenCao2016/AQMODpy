# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 10:37:05 2016

Constellation package 

including: 1. constellation operation
           2. block interleaving

@author: Chen Cao, University of Louisville
"""

import numpy as np

class constellation(object):
    def __init__(self,user_defined_constellation):
        """
        user_defined_constellation: complex constellation symbols array, single dimension, 
                                    mapping from the index of array to the number in the array.
                                    e.x., map: b'01<->user_defined_constellation[b'01]
                                    ** plz only use the constellations, the size of which obeys 2^x
        """
        
        if user_defined_constellation=='default': #QPSK
            self.constellation=np.array([1+1j,-1+1j,-1-1j,1-1j],dtype=complex)/np.sqrt(2)
        else:
            self.constellation=user_defined_constellation
            
           
        self.num_sym=len(self.constellation)        # number of symbols in the constellation 
        
        self.num_bits=np.log2(self.num_sym).astype(int) # number of bits in a constellation symbol
        
        
        self.constel_map=np.zeros((2,self.num_bits,self.num_sym/2),dtype=int)
        for i in xrange(self.num_bits):
            self.constel_map[0,i,:]=[x for x in xrange(self.num_sym) if (x>>i)%2==0]
            self.constel_map[1,i,:]=[x for x in xrange(self.num_sym) if (x>>i)%2==1]            

            
    def soft_demapping(self,mu_sym,delta_sym):
        """
        map constellation symbols to bits' LLR
        
        input: 
        symbol expectation mu_sym, dimension k*nt
        symbol variance delta_sym, dimension k*nt 
        
        output:
        bit_LLR, dimension (k*num_bits)*nt
        
        """
        k=int(mu_sym.shape[0])
        nt=int(mu_sym.shape[1])
        bit_LLR=np.zeros((k*self.num_bits,nt),dtype=float)
        mu_sym_extand=np.zeros((self.num_sym))[:,None,None]+mu_sym[None,:,:]
        delta_sym_extand=np.zeros((self.num_sym))[:,None,None]+delta_sym[None,:,:]
        constell_extand=self.constellation[:,None,None]+np.zeros((k,nt))[None,:,:]
        
        temp_variable=-np.abs(constell_extand-mu_sym_extand)**2/(delta_sym_extand)/2

        Pr_constell=np.exp(temp_variable)
        Pr_constell_sum=np.sum(Pr_constell,axis=0) 
        
        # deal with float point problem that may approximates to zero
        count=1
        while np.any(Pr_constell_sum<10**-300):
            temp_variable+=(np.array(Pr_constell_sum<10**-300).astype(int)[None,:,:]*100*count+np.zeros((self.num_sym))[:,None,None])            
            Pr_constell=np.exp(temp_variable)
            Pr_constell_sum=np.sum(Pr_constell,axis=0)
            count+=1            
            
        Pr_constell=Pr_constell/(Pr_constell_sum[None,:,:]+np.zeros((self.num_sym))[:,None,None]) #normalize
        
        for i in xrange(self.num_bits): 
            '''
            mu_sym_extand=np.zeros((self.num_sym/2))[:,None,None]+mu_sym[None,:,:]
            delta_sym_extand=np.zeros((self.num_sym/2))[:,None,None]+delta_sym[None,:,:]
            
            map_extand=self.constel_map[0,i,:][:,None,None]+np.zeros((k,nt))[None,:,:]        
            Pr_zero=norm.pdf((np.abs(map_extand-mu_sym_extand)**2).astype(float),loc=0,scale=delta_sym_extand)
            
            map_extand=self.constel_map[1,i,:][:,None,None]+np.zeros((k,nt))[None,:,:]
            Pr_one=norm.pdf((np.abs(map_extand-mu_sym_extand)**2).astype(float),loc=0,scale=delta_sym_extand)
            '''
            Pr_one=Pr_constell[self.constel_map[1,i,:],:,:]
            Pr_zero=Pr_constell[self.constel_map[0,i,:],:,:]
            sum_Pr_one=np.sum(Pr_one,axis=0)
            sum_Pr_zero=np.sum(Pr_zero,axis=0)
            bit_LLR[xrange(i*k,i*k+k),:]=np.log(sum_Pr_zero/sum_Pr_one)
            
            #bit_LLR[np.isnan(bit_LLR)]=0 #deal with log(0/0)
        if np.any(np.isnan(bit_LLR)):
            raise NameError('NaN in soft demapping, decoding is sufficient to stop') 
        return bit_LLR
        
    def soft_mapping(self,bit_LLR):
        """
        map bits' LLR to the contellation symbols 
        
        input:
        bit_LLR, dimension (k*num_bits)*nt
        
        output:
        symbol expectation mu_sym, dimension k*nt
        symbol variance delta_sym, dimension k*nt
        """        
        
        k=int(bit_LLR.shape[0])/self.num_bits
        nt=int(bit_LLR.shape[1])
        
        bit_LLR[np.isinf(bit_LLR)]=200   #deal with exp(inf),use arbitrary large number:100    
        
        Pr_zero=np.exp(bit_LLR)
        Pr_one=np.ones(bit_LLR.shape)
        temp=Pr_zero+Pr_one
        Pr_zero=Pr_zero/temp
        Pr_one=Pr_one/temp
        
        # change dimension to: num_bits*k*nt
        Pr_zero=Pr_zero.reshape((self.num_bits,k,nt))
        Pr_one=Pr_one.reshape((self.num_bits,k,nt))
        
        cache={0:Pr_zero,1:Pr_one}
        Pr_sym=np.ones((self.num_sym,k,nt))
        for x in xrange(self.num_sym):
            for i in xrange(self.num_bits):
                Pr_sym[x,:,:]*=cache[(x>>i)%2][i,:,:]
         
        #normalize the probability distribution
        Pr_sym=Pr_sym/(np.sum(Pr_sym,axis=0)[None,...]+np.zeros(self.num_sym)[:,None,None])

        mu_sym=np.sum([self.constellation[x]*Pr_sym[x,...] for x in xrange(self.num_sym)],axis=0)
        
        delta_sym=np.sum([np.abs(self.constellation[x]-mu_sym)**2 *Pr_sym[x,...] for x in xrange(self.num_sym)],axis=0)
        

        return mu_sym,delta_sym
        
        
        
        
        
class interleaver(object):
    def __init__(self,length,seed):
        """
        length: length of block interleaver
        seed: random seed
        """               
        rand_gen=np.random.mtrand.RandomState(seed)
        self.m_array=rand_gen.permutation(np.arange(length))
        
        
    def interleave(self,input_array):
        if len(input_array.shape)==1:
            interleaved = np.array(map(lambda x: input_array[x], self.m_array))
        else:
            nt=int(input_array.shape[1])
            interleaved=np.zeros(input_array.shape,input_array.dtype)
            for i in xrange(nt):
                interleaved[:,i]=np.array(map(lambda x: input_array[x,i], self.m_array))
        return interleaved 
        
    def deinterleave(self,input_array):
        deinterleaved = np.zeros(input_array.shape, input_array.dtype)
        if len(input_array.shape)==1:
            for index, element in enumerate(self.m_array):
                deinterleaved[element] = input_array[index]
        else:
            nt=int(input_array.shape[1])
            for i in xrange(nt):
                for index, element in enumerate(self.m_array):
                    deinterleaved[element,i] = input_array[index,i]                        
        return deinterleaved
        