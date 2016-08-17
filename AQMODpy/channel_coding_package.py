# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:09:10 2016

Channel coding package

including: LDCP coding (standard sum-product decoder)

@author: Chen Cao, University of Louisville
"""
import numpy as np

class LDPC (object):
    def __init__(self,H,G):
        self.G=G # generator matrix
        self.H=H # parity check matrix
        self.nf,self.nv=H.shape # size of H
        
        # factor graph relationship
        self.f_neighbor=[[a for a in range(self.nv) if H[i,a] ] for i in range(self.nf)]
        self.v_neighbor=[[a for a in range(self.nf) if H[a,i] ] for i in range(self.nv)]
        
    def encoder(self,in_bits):
        """
        LDPC encoder
        Plz check in_bit is binary (int32)
        
        input: 
        in_bits: source bit, single dimension array 
        
        output: 
        encoded bit, single dimension array
        """
        return in_bits.dot(self.G)%2
        
    def decoder(self,in_LLR,num_iter):
        """
        standard sum-product LDPC decoder
        
        input:
        in_LLR: bits LLR, single dimension array (float)
        num_iter: number of message-passing iterations
        
        output:
        L_posteriori: posteriori bits LLR                
        """        
        
        Lc = in_LLR

        Lq=np.zeros((self.nf,self.nv))
        Lr=np.zeros((self.nf,self.nv))
        
        count=0
             	
        while(True):
    
            count+=1    
           
            for i in xrange(self.nf):
                Ni = self.f_neighbor[i]
                for j in Ni:
                    Nij = Ni[:]   
                    if j in Nij: Nij.remove(j)                
                    if count==1:
                        X =np.prod(np.tanh(0.5*Lc[Nij]))
                    else:
                        X =np.prod(np.tanh(0.5*Lq[i,Nij]))

                    Lr[i,j] =2*np.arctanh(X)
            
            for j in xrange(self.nv):
                Mj = self.v_neighbor[j]                
                for i in Mj:
                    Mji = Mj[:]
                    if i in Mji: Mji.remove(i)    
                    Lq[i,j] = Lc[j]+np.sum(Lr[Mji,j])
            
                
            L_posteriori = np.zeros(self.nv)
            for j in xrange(self.nv):
                Mj = self.v_neighbor[j]    
                L_posteriori[j] = Lc[j] +np.sum(Lr[Mj,j])
                
            if count >= num_iter:  
                break
            
        return L_posteriori