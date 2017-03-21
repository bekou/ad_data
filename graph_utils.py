import utilities,scipy,file_parsers
import networkx as nx
import matrix_tree_theorem as mtt
import numpy as np

def GraphsFromGoldFile(file):
    
    #create gold tree
    # input: numpy array cols 'arg','left_id','right_id','rel_type','left_mention','right_mention'
    # output: list of document graphs
    gold_tree_graphs=[]
    goldFileDocs=file_parsers.goldFileParser(file).gold_docs
    for i in range (len(goldFileDocs)):#for all docs
        dG = nx.DiGraph()
        for j in range (len(goldFileDocs[i].left_id)):#for all lines in doc 
            left_node=str(goldFileDocs[i].left_id[j])+"/"+goldFileDocs[i].left_mention[j]
            right_node=str(goldFileDocs[i].right_id[j])+"/"+goldFileDocs[i].right_mention[j]
            
            if not dG.has_node(left_node):
                                        dG.add_node(left_node)
            if not dG.has_node(right_node ):
                                        dG.add_node(right_node )
            if not dG.has_edge(left_node, right_node):
                                        dG.add_edge(left_node, right_node, weight="1" )  
                    
        graph=graphDoc(goldFileDocs[i].docId,goldFileDocs[i].incrementalId,dG)
        gold_tree_graphs.append(graph)
      
    return gold_tree_graphs
    
def queryRelation(parent,child,parentType,childType,vectorizer,clf):
    return clf.decision_function(vectorizer.transform([featurePreprocessing(parent,child,parentType,childType)]))  

def queryFeature(feature,clf,vectorizer):
    return clf.decision_function(vectorizer.transform([feature]))  
    
class graphDoc:
    def __init__(self,id,iid,graph):
        self.incrementalId=iid
        self.docId=id
        self.graph=graph
    
def writeDocsGraphsToFile(node_file,graphStructure,output_file):

    #transform edmonds tree graph to file of the form e.g., RELATION 0 1 part-of ROOT PERCEEL
    # input names of the form 0/ROOT, 1/PERCEEL
    
    node_docs=file_parsers.nodeParser(node_file).node_docs
    
    writeList=[]
    print ("Writing the tree graph to a file...")
    for i in range (len(graphStructure)):
        print ".",
        line=graphStructure[i].docId
        writeList.append(line+"\n")
        docGraph=graphStructure[i]
        node_doc=node_docs[i]


        for l_id in range(len(node_doc.mention)): # iterate over all nodes - to create the left side            
            for r_id in range(len(node_doc.mention)):                
                for j in range (len(docGraph.graph.edges())): 

                    left=docGraph.graph.edges()[j][0]
                    left_id=left.split("/")[0]            
                    right=docGraph.graph.edges()[j][1]
                    right_id=right.split("/")[0]
                    
                    left_mn=node_doc.mention[int(left_id)]
                    right_mn=node_doc.mention[int(right_id)]


                    if l_id==int(left_id) and r_id==int(right_id):                    
            
                
                            line="RELATION"+"\t"+left_id+"\t"+right_id+"\t"+"part-of"+"\t"+left_mn+"\t"+right_mn
                            writeList.append(line+"\n")
        
    print ""
    utilities.writeListToFile(output_file,writeList)

def EdmondGraphs(graphs,path=""):
    # input: graph as a list of graphs
    edmond_graphs=[]
    print ("Transforming the relation file to a tree...")
    for i in range (len(graphs)):
        print ".",
        try:
            min=nx.maximum_spanning_arborescence(graphs[i].graph, attr='weight', default=1)
        except Exception as e :
            print  e
            print graphs[i].docId
            try :
                draw_func_all_windows(graphs[i].graph,path+" "+ str(graphs[i].docId)+".png",True)   
            except UnicodeEncodeError as x:
                print x
        graph=graphDoc(graphs[i].docId,graphs[i].incrementalId,min)
        edmond_graphs.append(graph)
    print ""
    return edmond_graphs    
    
def getGraphsGivenMttTheta(thetadocs,root_thetadocs,node_docs):       
        
        graphs=[]

        for doc in range(len(node_docs)):
            dG = nx.DiGraph()    

            theta=thetadocs[doc]
            root_theta=root_thetadocs[doc]
            nodeList=node_docs[doc]
            rootIndex=nodeList.mention.index("ROOT")
            #nodes=len(nodeList.mention)-1
            for h in range(len(nodeList.mention)):
                left=nodeList.mention[h]
                for m in range(len(nodeList.mention)):
                    
                    if h!=m and m!=rootIndex:
                        right=nodeList.mention[m]
                        w=-1
                                    
                        w=theta[h,m]
                       
                        left_node=str(h)+"||"+left
                        right_node=str(m)+"||"+right
                            
                        if not dG.has_edge(left_node, right_node):
                                            dG.add_edge(left_node, right_node, weight=w) 
            
            
            graphs.append(nx.maximum_spanning_arborescence(dG))
            
        return graphs
            


def getPredictions(node_docs,graph):
    
    y_edmonds=[]
    
    for i in range (len(node_docs)):
        
        nodeDoc=node_docs[i]
        
        graphDoc=graph[i]
        
        rootIndex=nodeDoc.mention.index("ROOT")
        
        ptr=0;
        for j in range(len(nodeDoc.mention)):
            left_node_mn=nodeDoc.mention[j]
            
           
            for z in range(len(nodeDoc.mention)):
                right_node_mn=nodeDoc.mention[z]
                
                rel=""
                
                if (j!=z and z!=rootIndex):

                    label=0
                    for ed in range (len(graphDoc.edges())):
                
                            left=graphDoc.edges()[ed][0].split("||")
                            right=graphDoc.edges()[ed][1].split("||")  
                            
                            left_id=int(left[0])
                            right_id=int (right[0])
                           
                            if (j==left_id and z==right_id):
                                      label=1                          
                                
                    
                    y_edmonds.append(label)
                    
            
    return y_edmonds

    
    
def weightedGraphsFromFeatures(rel_docs,node_docs,clf,vectorizer):
   
    
    graphs=[]   
    print "Creating graphs from feature file..."
    for i in range (len(node_docs)):
        print ".",
        dG = nx.DiGraph()
        nodeDoc=node_docs[i]
        relDoc=rel_docs[i]   
        
        
        
        rootIndex=nodeDoc.mention.index("ROOT")
        
        
        ptr=0;
        for j in range(len(nodeDoc.mention)):
            left_node_mn=nodeDoc.mention[j]
             
            
           
            for z in range(len(nodeDoc.mention)):
                right_node_mn=nodeDoc.mention[z]
                
                rel=""
                
                if (j!=z and z!=rootIndex):
                    left_side=str(j)+ "/" + left_node_mn
                    right_side=str(z)+ "/" + right_node_mn
                    
                    score=queryFeature(relDoc.lines[ptr],clf,vectorizer)[0]
                    
                    
                    
                    if not dG.has_node(left_side):
                                            dG.add_node(left_side)
                    if not dG.has_node(right_side ):
                                            dG.add_node(right_side )
                    if not dG.has_edge(left_side, right_side):
                                            dG.add_edge(left_side, right_side, weight=score)   
                                            
                    ptr+=1
        graph=graphDoc(relDoc.docId,relDoc.incrementalId,dG)   
        graphs.append(graph)
    print ""            
    return graphs  
    
    
def writeRelationsFile(features_file,predictions,outFile): # write relation file given the feature file 
    import copy
    rel_docs=featuresFileParser(features_file).feature_docs#[0].right_mention
    pred_docs = copy.copy(rel_docs)
    for i in range (len(pred_docs)):
        for j in range (len(pred_docs[i].label)):
            pred_docs[i].label[j]=0
    
    
    ptr=0
    for i in range (len(pred_docs)):       
        predDoc=pred_docs[i]
        
        for j in range (len(predDoc.label)):
            
            predDoc.label[j]=predictions[ptr]
            
            ptr+=1
            
    writeList=[]
    for i in range (len(pred_docs)):
        predDoc=pred_docs[i]
        line=predDoc.docId
        writeList.append(line+"\n")
        for j in range (len(predDoc.label)):            
            line=str(pred_docs[i].label[j])+"\t" +pred_docs[i].lines[j]
            writeList.append(line+"\n")
    writeListToFile(outFile,writeList)
            
    writeListToFile(outFile,writeList)



def getPredictionsFromEdmond(rel_docs,node_docs,graph,clf,vectorizer):
    
   
    y_edmonds_train=[]
    
    print "Get Predictions From Edmond graph" 
    for i in range (len(node_docs)):
        
        print ".",
        nodeDoc=node_docs[i]
        relDoc=rel_docs[i]   
        graphDoc=graph[i]
        
        rootIndex=nodeDoc.mention.index("ROOT")
        
        
        ptr=0;
        for j in range(len(nodeDoc.mention)):
            left_node_mn=nodeDoc.mention[j]
            
            
           
            for z in range(len(nodeDoc.mention)):
                right_node_mn=nodeDoc.mention[z]
                 
                rel=""
                
                if (j!=z and z!=rootIndex):
                    label=0
                    for ed in range (len(graphDoc.graph.edges())):
                
                            left=graphDoc.graph.edges()[ed][0].split("/")
                            right=graphDoc.graph.edges()[ed][1].split("/")  
                            
                            left_id=int(left[0])
                            right_id=int (right[0])
                             
                            if (j==left_id and z==right_id):
                                      label=1                          
                                
                    ptr+=1               
                    y_edmonds_train.append(label)
                    
    print ""        
    return y_edmonds_train
    
def getMTTLabels(X_train,X_test,node_docs_train,node_docs_test,y_train,c): # run the MTT pipeline

            w=np.zeros((1,X_train.shape[1]))

            labels_sparse = scipy.sparse.csr_matrix(map(int, y_train)) 
                            
            featuresActivated=labels_sparse.T.multiply(X_train)
                            
            featuresSum=scipy.sparse.csr_matrix.sum(featuresActivated,axis=0)
                                                       
            myargs = (X_train,node_docs_train)

            x,f,d=scipy.optimize.lbfgsb.fmin_l_bfgs_b(mtt.L,x0=w,fprime=mtt.gradL,args=(X_train,node_docs_train,y_train,c,featuresSum),iprint=1,maxiter=1000)
                
            theta_test=mtt.computeTheta(np.matrix(x),X_test,node_docs_test)
            theta_test_doc=theta_test[0]
            root_theta_test_doc=theta_test[1]

            mtx=mtt.computeMtx(X_test,node_docs_test,theta_test_doc,root_theta_test_doc)  
            adjacency_doc=mtx[0]
            laplacian_doc=mtx[1]
            partitionLog=mtx[2]


            test_graphs=getGraphsGivenMttTheta(theta_test_doc,root_theta_test_doc,node_docs_test)    

            pred_labels=getPredictions(node_docs_test,test_graphs)  
            
            
            return pred_labels