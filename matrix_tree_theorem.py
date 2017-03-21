import networkx as nx
import numpy as np
import math
import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib
from time import gmtime, strftime
import scipy

margFeat=[]

def getAdjacency(theta):
    adjacency=np.exp(theta)
    #print "adjacency max "+str(np.amax(adjacency))
    np.fill_diagonal(adjacency,0)
    return adjacency

def getmLaplacian(adjacency,root_theta):
    
        laplacian=-adjacency
        for i in range (len(laplacian)):
            
            laplacian[i,i]=sum(adjacency[:,i])[0,0]
        
        z=np.zeros((root_theta.shape[1],root_theta.shape[1]))
        np.fill_diagonal(z,np.exp(root_theta))
        
        laplacian[0]=np.exp(root_theta)
        return laplacian
    
def getMarginal(laplacian,adjacency,root_theta):

    delta=np.zeros((len(laplacian),len(laplacian)))
    np.fill_diagonal(delta,1)
    inv_laplacian=np.linalg.inv(laplacian)
    marg_p=np.zeros((len(laplacian),len(laplacian)))
    for h in range (len(laplacian)):    
        for m in range (len(laplacian)):
            #print str(h) + " " + str(m )
            marg_p[h,m]=(1-delta[0,m])*adjacency[h,m]*inv_laplacian[m,m]- (1-delta[h,0])*adjacency[h,m]*inv_laplacian[m,h]
    
    
    root_marg=np.zeros((1,len(laplacian)))
    for m in range (len(laplacian)):    
        root_marg[0,m]=np.exp(root_theta[0,m])*inv_laplacian[m,0]
      
    return marg_p,root_marg

def computeTheta(w,train_vec,node_docs,labels=""):      
        
        theta_list=(train_vec*w.T)
        
        ptr=0;
        theta_doc=[]
        root_theta_doc=[]
        theta_active_sum=[]
        
        for doc in range (len(node_docs)):

                    theta_active_sum.append(0)
                    nodeDoc=node_docs[doc]
                    rootIndex=nodeDoc.mention.index("ROOT")
                    nodes=len(nodeDoc.mention)


                    thetas=np.asmatrix(np.zeros((nodes,nodes)))
                    root_thetas=np.asmatrix(np.zeros((1,nodes)))
                    for h in range(len(nodeDoc.mention)):
                       

                        for m in range(len(nodeDoc.mention)):
                            

                            if (h!=m and m!=rootIndex):
                                    
                                    thetas[h,m]=theta_list[ptr]
                                     

                                    if labels!="" and labels[ptr]=='1':
                                       
                                        theta_active_sum[doc]+=theta_list[ptr]
                                        
                                        
                                    ptr+=1
                   
                    root_thetas=np.zeros((1,len(thetas)))
                    
                    theta_doc.append(thetas)
                    root_theta_doc.append(root_thetas)
                    
        return theta_doc,root_theta_doc,theta_active_sum,theta_list



def computeMtx(train_vec,node_docs,theta_doc,root_theta_doc):

    
        adjacency_doc=[]
        laplacian_doc=[]
        partitionLog=[]
        marginal_doc=[]
        root_marg_doc=[]
        for doc in range (len(theta_doc)):
    
            adjacency_doc.append(getAdjacency(theta_doc[doc]))
            
            laplacian_doc.append(getmLaplacian(adjacency_doc[doc],root_theta_doc[doc]))
            
            (sign, logdet) = np.linalg.slogdet(laplacian_doc[doc])
            
            partitionLog.append(sign*logdet)

            #partitionLog.append(np.log(np.linalg.det(laplacian_doc[doc])))
            
            marg=getMarginal(laplacian_doc[doc],adjacency_doc[doc],root_theta_doc[doc])

            marginal_doc.append(marg[0])
            
            root_marg_doc.append(marg[1])
            
            
        ptr=0;
        margs=[]
        

        for doc in range (len(marginal_doc)):
    
            marginal=marginal_doc[doc]
            root_marg=root_marg_doc[doc]
            for h in range(len(marginal)):
                          
                for m in range(len(marginal)):
                    marg=marginal[h,m]
                    
                    if (h!=m and m<len(marginal)-1):
                            
                            margs.append(marg)
                            ptr+=1
                            
           
        margs=np.asarray(margs)
        
        margs = scipy.sparse.csr_matrix(margs) 
        
        margFeat=margs.T.multiply(train_vec)
        
        return adjacency_doc,laplacian_doc,partitionLog,marginal_doc,root_marg_doc,margFeat

def printTime(messageBefore):
                print messageBefore+" : "+strftime("%Y-%m-%d %H:%M:%S")    

    
def L(w,train_vec,node_docs,labels,C,featuresSum):
    
   
    w=np.matrix(w)
    
    
    theta=computeTheta(w,train_vec,node_docs,labels)
    
    theta_doc=theta[0]
    root_theta_doc=theta[1]
    theta_active_sum=theta[2]
    logSum=0
    mtx=computeMtx(train_vec,node_docs,theta_doc,root_theta_doc)  
    

    adjacency_doc=mtx[0]
    laplacian_doc=mtx[1]
    partitionLog=mtx[2]
    marginal_doc=mtx[3]
    root_marg_doc=mtx[4]
    global margFeat
    margFeat=mtx[5]
    for doc in range (len(theta_doc)):
                
                logSum+=theta_active_sum[doc]-partitionLog[doc]
                

    L=-C*logSum+0.5*math.pow(np.linalg.norm(w),2)

    if logSum>0:
	raise ValueError('--Log likelihood is positive --')

    print "----------------------------------------------------------"
    print "Objective: "+str(L)+ " Likelihood: " + str(logSum)
    print "----------------------------------------------------------"
    

    return L

def gradL(w,train_vec,node_docs,labels,C,featuresSum):
        

        w=np.matrix(w)
        

        sumMargFeat=np.zeros((1,train_vec.shape[1])).astype('double')   
        
        

        sumMargFeat=scipy.sparse.csr_matrix.sum(margFeat,axis=0)#np.sum(margFeat,axis=0)
        
        dL=w-C*featuresSum+C*sumMargFeat
        dL=np.squeeze(np.asarray(dL))
        
        
        return dL
    
   

    
