import matplotlib.pyplot as plt
#%matplotlib inline
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from unidecode import unidecode
from sklearn.cross_validation import train_test_split
import os
import errno
import re
import codecs
import matrix_tree_theorem as mtt
import scipy.optimize.lbfgsb as lbfgs
import scipy
from pandas_confusion import ConfusionMatrix,BinaryConfusionMatrix



#functions for fixing unicode issues  
def remove_non_ascii(text):
    return unidecode(unicode(text, encoding = "utf-8"))
def fixUnicode(text):
      return str(text).decode('ascii', 'ignore')


def GraphsFromGoldFile(file):
    
    #create gold tree
    # input: numpy array cols 'arg','left_id','right_id','rel_type','left_mention','right_mention'
    # output: list of document graphs
    gold_tree_graphs=[]
    goldFileDocs=goldFileParser(file).gold_docs
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

def writeDocsGraphsToFile(node_file,graphStructure,output_file):

    #transform edmonds tree graph to file of the form e.g., RELATION 0 1 part-of ROOT PERCEEL
    # input names of the form 0/ROOT, 1/PERCEEL
    
    node_docs=nodeParser(node_file).node_docs
    
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
    writeListToFile(output_file,writeList)

def inOrder(graph,root,traversal):
        # in order traversal of a graph for the transition based system
        if (root != None):
            
                children=graph.successors(root)
                leftChildren=[]
                rightChildren=[]
                for child in children:
                    rootId=root.split("/")[0]
                    childId=child.split("/")[0]
                    if rootId>childId:
                        leftChildren.append(child)
                    elif rootId<childId:
                        rightChildren.append(child)
                 
                
                leftChildren=sortNodeList(leftChildren)
                rightChildren=sortNodeList(rightChildren)
                
                #print "node " + root + " left " + str(leftChildren)
                #print "node " + root + " right " + str(rightChildren)

                for left in leftChildren:
                   
                    inOrder(graph,left,traversal);
                    
                    
                #Visit the node by Printing the node data
                #print root
                traversal.append(root)
                #print root
                
                for right in rightChildren:
                    inOrder(graph,right,traversal);
                    
        return traversal

def sortNodeList(list):
    # sort the node list based on the id of the node
    # return the sorted list
    sortedList=[]
    if len(list)>0:
        ids=[]
        
        for l in list:
                ids.append(int(l.split("/")[0]))
        maxId=max(ids)
        for i in range(maxId+1):
            for l in list:
                id=int(l.split("/")[0])
                if i==id :
                    sortedList.append(l)
    return sortedList
    
    
class graphObject:
    # structure that holds the transition based graph after an inorder traversal
    def __init__(self,name,nodeId,traversedId,nodeName,type):
            self.name=name
            self.nodeId=int(nodeId)
            self.traversedId=int(traversedId)
            self.nodeName=nodeName
            self.type=type

class Oracle:
    
        def __init__(self,graphObjects,graph):
                # the oracle with the swap operation inside

                mch=machine()

                mch.stack.append(graphObjects[0])
                mch.bufferL=graphObjects[1:]

                self.stacks=[]
                self.buffers=[]
                self.actions=[]
                
                self.arcs=[]

                self.arcs.append(list(mch.arcs))
                self.stacks.append(list(mch.stack))
                self.buffers.append(list(mch.bufferL))

                nodeDependents=[]
                nodeNames=[]
                for g in graphObjects:
                    #print g.nodeName
                    nodeNames.append(g.nodeName)
                    dependentsOfNode=graph.graph.successors(g.nodeName)
                    nodeDependents.append(dependentsOfNode)
                self.iterations=0

                
                while (mch.isNotInEndState()):

                    top_2=mch.stack[len(mch.stack)-2]
                    top_1=mch.stack[len(mch.stack)-1]
                    idxHead=nodeNames.index(top_2.nodeName)
                    idxTail=nodeNames.index(top_1.nodeName)
                    projOrderHead=getIndexGivenTravId(graphObjects,idxHead)
                    projOrderTail=getIndexGivenTravId(graphObjects,idxTail)
                    dependent=None
                    
                    # add left or right relation if there are no dependents
                    if graph.graph.has_edge(top_2.nodeName,top_1.nodeName) and top_1.nodeId!=0:
                        side="right"
                        head=top_2
                        dependent=top_1
                        
                    elif graph.graph.has_edge(top_1.nodeName,top_2.nodeName) and top_2.nodeId!=0: 
                        side= "left"
                        head=top_1
                        dependent=top_2
                       
                    action=None
                    if dependent!=None and len(nodeDependents[nodeNames.index(dependent.nodeName)])==0:# check if it has no dependents left 
                       
                            if side=="left":
                                mch.leftArc()
                                action= "left"
                            elif side=="right":
                                mch.rightArc()
                                action= "right"

                            idxHead = nodeNames.index(head.nodeName)

                            nodeDependents[idxHead].remove(dependent.nodeName)  # remove the dependent from the head dependents' list

                    elif top_2.nodeId>0 and top_2.nodeId<top_1.nodeId and projOrderHead>projOrderTail:

                        mch.swap()
                        action= "swap"

                    elif len(mch.bufferL)>0:

                        mch.shift()
                        action= "shift"

                    
                    self.actions.append(action)
                    self.stacks.append(list(mch.stack))
                    self.buffers.append(list(mch.bufferL))
                    self.arcs.append(list(mch.arcs))
                    self.iterations+=1
                    
                    
class machine:
    # Machine actions
    def __init__(self):
            self.stack=[]
            self.bufferL=[]
            self.arcs=[]
            self.actions=[]
            self.parent=None
            self.ds=0
            self.gradientAdded=False
            self.isCorrect=None
            self.wasCorrect=None
        
    def swap(self):
        top_2=self.stack[len(self.stack)-2]
        self.stack.remove(top_2)
        self.bufferL=[top_2]+self.bufferL
        self.actions.append("swap")
        
        
    def shift(self):
        
        top=self.bufferL[0]
        self.bufferL.remove(top)
        self.stack.append(top)
        self.actions.append("shift")
         

        
    def leftArc(self):
        
        top_2=self.stack[len(self.stack)-2]
        top_1=self.stack[len(self.stack)-1]
        self.arcs.append(arcObject(top_1,"part-of",top_2,"left_arc"))
        self.stack.remove(top_2)
        self.actions.append("left")
         

        
        
    def rightArc(self):
        
        top_2=self.stack[len(self.stack)-2]
        top_1=self.stack[len(self.stack)-1]
        self.arcs.append(arcObject(top_2,"part-of",top_1,"right_arc"))
        self.stack.remove(top_1)
        self.actions.append("right")
        
    
    def isNotInEndState(self):
        
        return len(self.bufferL)>0 or len(self.stack)!=1
    
    def getRandomAction(self):
    
        import random
        intAction=random.randint(0,3)

        if intAction==0:
            action="left"
        elif intAction==1:
            action="right"
        elif intAction==2:
            action="shift"
        elif intAction==3:
            action="swap"

        return action
    
    def isPermissible(self,action):

        top_2=self.stack[len(self.stack)-2]
        top_1=self.stack[len(self.stack)-1]

        if action=="left"  and top_2.nodeId!=0: #and top_1.nodeId>top_2.nodeId:
            
            return True
        elif action=="right" and top_1.nodeId!=0: #and top_1.nodeId>top_2.nodeId:
            
            return True
        elif action=="swap" and top_2.nodeId>0 and top_2.nodeId<top_1.nodeId:
            
            return True
        elif action=="shift" and len(self.bufferL)>0 :
           
            return True
        return False

    def getPermissibleActions(self):

        actions=["left","right","swap","shift"]
        permissibleActions=[]
        for action in actions:
                if self.isPermissible(action):
                    permissibleActions.append(action)
                    
        return permissibleActions
        
# features used in the transition based parser        
class featurePreprocessingTransitions:
    def __init__(self):
        self.s0_type=True 
        self.s1_type=True 
        
        self.s0_word=True 
        self.s1_word=True 
        
        self.b0_type=True 
        self.b1_type=True 
        self.b2_type=False 
        self.b3_type=False 
        
        self.b0_word=True 
        self.b1_word=True 
        self.b2_word=False 
        self.b3_word=False 
        
        self.a0_word=False 
        
        
        self.left_s0_word=True 
        self.right_s0_word=True 
        
        self.left_s0_type=True 
        self.right_s0_type=True 
        
        
        self.s0_successors=False
        
        self.s0_b0_word=True        
        self.s0_s1_word=True        
        self.b0_b1_word=True
        
        
        self.s0_b0_type=True        
        self.s0_s1_type=True        
        self.b0_b1_type=True
        
        
        self.s1_b0_word=True
        self.s1_b0_type=True
        
        self.s1_s0_b0_word=False
        self.s0_b0_b1_word=False
        self.b0_b1_b2_word=False
        self.b1_b2_b3_word=False
        
        
    def getLeftRightMost(self,arcs,graph,node,graphObject):
        
        
        successors=[]
        try:
            successors=graph.successors(node.nodeName)
        except nx.exception.NetworkXError as e :
            
            pass
         
        
        sortedSuccessors=sortNodeList(successors)
        

        leftW=""
        rightW=""
        leftT=""
        rightT=""
        if len(sortedSuccessors)>0:
               
                
                
                for gIdx in range(len(graphObject)):
                
                        if sortedSuccessors[0]==graphObject[gIdx].nodeName:
                           leftW=graphObject[gIdx].name
                           leftT=graphObject[gIdx].type

                           
                        if sortedSuccessors[len(sortedSuccessors)-1]==graphObject[gIdx].nodeName:
                           rightW=graphObject[gIdx].name
                           rightT=graphObject[gIdx].type

                
        
        return leftW,rightW,leftT,rightT,sortedSuccessors
        
        
        
        
        
        
        
             
    def process(self,stack,bufferL,arcs,graphObject):
       
    
        features=""
        
        s0_typeTAG="s0_type#|"
        s1_typeTAG="s1_type#|" 
        
        s0_wordTAG="s0_word#|" 
        s1_wordTAG="s1_word#|" 
        
        b0_typeTAG="b0_type#|" 
        b1_typeTAG="b1_type#|" 
        b2_typeTAG="b2_type#|" 
        b3_typeTAG="b3_type#|" 
        
        b0_wordTAG="b0_word#|" 
        b1_wordTAG="b1_word#|" 
        b2_wordTAG="b2_word#|" 
        b3_wordTAG="b3_word#|" 
        
        a0_wordTAG="a0_word#|" 
        
        leftMost_s0_word_TAG="leftMost_s0_word#|"        
        rightMost_s0_word_TAG="rightMost_s0_word#|"
        
        leftMost_s0_type_TAG="leftMost_s0_type#|"        
        rightMost_s0_type_TAG="rightMost_s0_type#|"
        
        
        s0_successors_TAG="s0_successors_word#|"
        
        
        s0_b0_word_TAG="s0_b0_word#|"        
        s0_s1_word_TAG="s0_s1_word#|"        
        b0_b1_word_TAG="b0_b1_word#|"
        
        
        s0_b0_type_TAG="s0_b0_type#|"        
        s0_s1_type_TAG="s0_s1_type#|"        
        b0_b1_type_TAG="b0_b1_type#|"
        
        s1_b0_word_TAG="s1_b0_word#|"   
        s1_b0_type_TAG="s1_b0_type#|"           


        s1_s0_b0_word_TAG="s1_s0_b0#|"
        s0_b0_b1_word_TAG="s0_b0_b1#|"
        b0_b1_b2_word_TAG="b0_b1_b2#|"
        b1_b2_b3_word_TAG="b1_b2_b3#|"      

        
        
        

        
        graph=graphFromArcObject(arcs)
        
        
        leftW,rightW,leftT,rightT,sortedSuccessors=self.getLeftRightMost(arcs,graph,stack[len(stack)-1],graphObject)  
        
       
        arc0=""
        for arcIdx in range(len(arcs)):
            
            if arcIdx==len(arcs)-1:
                arc0=arcs[arcIdx].left.name + "_" +arcs[arcIdx].right.name + "_" +arcs[arcIdx].transition
        
        
        if self.s0_type==True:
            feature_set_1=s0_typeTAG+replaceWhiteSpaces(stack[len(stack)-1].type)#s0_type
            features+=feature_set_1
        
        features+=" "
        
        
        if self.s1_type==True and len(stack)>1:
            feature_set_2=s1_typeTAG+replaceWhiteSpaces(stack[len(stack)-2].type)#s1_type
            features+=feature_set_2
        
        features+=" "
        
        
        if self.s0_word==True:
            feature_set_3=s0_wordTAG+replaceWhiteSpaces(stack[len(stack)-1].name)#s0_word
            features+=feature_set_3
        
        features+=" "
        
        
        if self.s1_word==True and len(stack)>1:
            feature_set_4=s1_wordTAG+replaceWhiteSpaces(stack[len(stack)-2].name)#s1_word
            features+=feature_set_4
        
        features+=" "
        
       
        
        if self.b0_type==True and len(bufferL)>0:
            feature_set_5=b0_typeTAG+replaceWhiteSpaces( bufferL[0].type)#b0_type
            features+=feature_set_5
        
        features+=" "
        
        if self.b0_word==True and len(bufferL)>0:
            feature_set_6=b0_wordTAG+replaceWhiteSpaces( bufferL[0].name)#b0_word
            features+=feature_set_6
        
        features+=" "
        
        
        if self.b1_type==True and len(bufferL)>1:
            feature_set_7=b1_typeTAG+replaceWhiteSpaces( bufferL[1].type)#b1_type
            features+=feature_set_7
        
        features+=" "
        
        if self.b1_word==True and len(bufferL)>1:
            feature_set_8=b1_wordTAG+replaceWhiteSpaces( bufferL[1].name)#b1_word
            features+=feature_set_8
        
        features+=" "
        
        
        if self.b2_type==True and len(bufferL)>2:
            feature_set_9=b2_typeTAG+replaceWhiteSpaces( bufferL[2].type)#b1_type
            features+=feature_set_9
        
        features+=" "
        
        if self.b2_word==True and len(bufferL)>2:
            feature_set_10=b2_wordTAG+replaceWhiteSpaces( bufferL[2].name)#b1_word
            features+=feature_set_10
        
        features+=" "
        
        if self.b3_type==True and len(bufferL)>3:
            feature_set_11=b3_typeTAG+replaceWhiteSpaces( bufferL[3].type)#b1_type
            features+=feature_set_11
        
        features+=" "
        
        if self.b3_word==True and len(bufferL)>3:
            feature_set_12=b3_wordTAG+replaceWhiteSpaces( bufferL[3].name)#b1_word
            features+=feature_set_12
        
        features+=" "
        
        
        if self.a0_word==True:
            feature_set_13=a0_wordTAG+replaceWhiteSpaces( arc0)#a0_word
            features+=feature_set_13
        
        features+=" "
        
        if self.left_s0_word==True:
            feature_set_14=leftMost_s0_word_TAG+replaceWhiteSpaces(leftW)#a0_word
            features+=feature_set_14
        
        features+=" "
        
        if self.right_s0_word==True:
            feature_set_15=rightMost_s0_word_TAG+replaceWhiteSpaces(rightW)#a0_word
            features+=feature_set_15
        
        features+=" "
        
        
        if self.s0_successors==True:
        
            for node in successors:
                node=''.join(node.split("/")[1:])
                features+=s0_successors_TAG+node
                features+=" "
        
        features+=" "
        
        
        if self.s0_b0_word==True :
            feature_set_16=s0_b0_word_TAG+replaceWhiteSpaces(stack[len(stack)-1].name)#a0_word
            
            if len(bufferL)>0:
            
                feature_set_16+="_"+replaceWhiteSpaces(bufferL[0].name)
            features+=feature_set_16
        
        features+=" "
        
        
        if self.s0_s1_word==True :
            feature_set_17=s0_s1_word_TAG+replaceWhiteSpaces(stack[len(stack)-1].name)#a0_word
            
            if len(stack)>1:
            
                feature_set_17+="_"+replaceWhiteSpaces(stack[len(stack)-2].name)
            features+=feature_set_17
        
        features+=" "
        
        
        if self.b0_b1_word==True :
            feature_set_18=b0_b1_word_TAG
            if len(bufferL)>0:
                feature_set_18+=replaceWhiteSpaces(bufferL[0].name)#a0_word
            
            if len(bufferL)>1:
            
                feature_set_18+="_"+replaceWhiteSpaces(bufferL[1].name)
            features+=feature_set_18
        
        features+=" "
        
        
        
        if self.s0_b0_type==True :
            feature_set_19=s0_b0_type_TAG+replaceWhiteSpaces(stack[len(stack)-1].type)#a0_type
            
            if len(bufferL)>0:
            
                feature_set_19+="_"+replaceWhiteSpaces(bufferL[0].type)
            features+=feature_set_19
        
        features+=" "
        
        
        if self.s0_s1_type==True :
            feature_set_20=s0_s1_type_TAG+replaceWhiteSpaces(stack[len(stack)-1].type)#a0_word
            
            if len(stack)>1:
            
                feature_set_20+="_"+replaceWhiteSpaces(stack[len(stack)-2].type)
            features+=feature_set_20
        
        features+=" "
        
        
        if self.b0_b1_type==True :
            feature_set_21=b0_b1_type_TAG
            if len(bufferL)>0:
                feature_set_21+=replaceWhiteSpaces(bufferL[0].type)#a0_word
            
            if len(bufferL)>1:
            
                feature_set_21+="_"+replaceWhiteSpaces(bufferL[1].type)
            features+=feature_set_21
        
        features+=" "
        
        
        if self.left_s0_type==True:
            feature_set_22=leftMost_s0_type_TAG+replaceWhiteSpaces(leftT)#a0_word
            features+=feature_set_22
        
        features+=" "
        
        if self.right_s0_type==True:
            feature_set_23=rightMost_s0_type_TAG+replaceWhiteSpaces(rightT)#a0_word
            features+=feature_set_23
        
        features+=" "
        
        
        if self.s1_b0_word==True :
            feature_set_24=s1_b0_word_TAG
            if len(stack)>1:
                feature_set_24+=replaceWhiteSpaces(stack[len(stack)-2].name)#a0_word
            
            if len(bufferL)>0:
            
                feature_set_24+="_"+replaceWhiteSpaces(bufferL[0].name)
            features+=feature_set_24
        
        features+=" "
        
        
        if self.s1_b0_type==True :
            feature_set_26=s1_b0_type_TAG
            if len(stack)>1:
                feature_set_26+=replaceWhiteSpaces(stack[len(stack)-2].type)#a0_word
            
            if len(bufferL)>0:
            
                feature_set_26+="_"+replaceWhiteSpaces(bufferL[0].type)
            features+=feature_set_26
        
        features+=" "
        
        
        
        
        if self.s1_s0_b0_word==True :
            feature_set_25=s1_s0_b0_word_TAG
            if len(stack)>1:
                feature_set_25+="s1_"+replaceWhiteSpaces(stack[len(stack)-2].name)#a0_word
                
            if len(stack)>0:
                feature_set_25+="s0_"+replaceWhiteSpaces(stack[len(stack)-2].name)#a0_word
            
            if len(bufferL)>0:
            
                feature_set_25+="b0_"+replaceWhiteSpaces(bufferL[0].name)
            features+=feature_set_25
        
        features+=" "
        
        return features  
        
def graphFromArcObject(arcs):

      graph_test=nx.DiGraph()
      for arc in arcs:
            
            graph_test.add_edge(arc.left.nodeName,arc.right.nodeName)
      
      return graph_test  
      
class arcObject:
    def __init__(self,left,label,right,transition):
            self.left=left
            self.label=label
            self.right=right
            self.transition=transition    
            
def getIndexGivenTravId(graphObjects,tId):
    for idx in range(len(graphObjects)):
        if graphObjects[idx].traversedId==tId:
            return idx
        
            
def parseGraphs(graph,traversal,node_docs):
    nodes=[]
    for node in graph.graph:
            nodes.append(node)
                    #print nodes

    

    nodes=sortNodeList(nodes)



    
    graphObjects=[]
    
    for i in range(len(nodes)):
        node=nodes[i]
        id=int(node.split("/")[0])
        name=node.split("/")[1]
        
        traversedId=int(traversal[i].split("/")[0])
        
        type=None
        for i in range (len(node_docs[graph.incrementalId].nodeId)):
                
                if int(node_docs[graph.incrementalId].nodeId[i])==id:
                    #print "mphka"
                    type=node_docs[graph.incrementalId].type[i]
                    #print type
        graphObjects.append(graphObject(name,id,traversedId,node,type))
        
    return graphObjects    


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

def writeListToFile(filename,lst):
    with codecs.open(filename, 'w',"utf-8") as f:
        for s in lst:
            
            f.write(s )

def appendListToFile(filename,lst):
            
    with codecs.open(filename, 'a',"utf-8") as f:
        for s in lst:
            
            f.write(s )
            
def computeDocIDs(file):
    doc_ids=[]
    for i in range (file.shape[0]):
        if '#doc' in file[i][0]:
                doc_ids.append(file[i][0])
    return doc_ids

def getDocID(ids,id):
    return [i for i,x in enumerate(ids) if id in x]

def getSeqIDFromDocID(ids,id):
    return ids[id]

def fileToList(file):
    doc=[]
    list_docs=[]
    
    for i in range (len(file)):

        if '#doc' in file[i][0] or i==file.shape[0]-1:
            if doc:
                list_docs.append(doc)           
                doc=[]       
        else: 
            doc.append(file[i])

    return list_docs

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


    
    
def createFeaturesFile(goldFile,nodeFile,outfile,tokenfile,featuresFile):

    # create graph based features given the relations 
    feat=featurePreprocessing(featuresFile)
    total_edges=0
    total_segments=0
    
    gold_docs=goldFileParser(goldFile).gold_docs
    node_docs=nodeParser(nodeFile).node_docs
    
    token_docs=tokenParser(tokenfile).token_docs

    writeList=[]
    
    print ("Computing features for the input relation file...")
    
    for i in range (len(node_docs)):
        #docWriteList=[]
        print ("."),
        nodeDoc=node_docs[i]
        tokenDoc=token_docs[i]
        goldDoc=gold_docs[i]
        
        
        line=nodeDoc.docId
        writeList.append(line+"\n")
        
        rootIndex=nodeDoc.mention.index("ROOT")
        
        
        edges=0
        total_segments+=len(nodeDoc.mention)
        
        for j in range(len(nodeDoc.mention)): # iterate over all nodes - to create the left side
            left_node_mn=nodeDoc.mention[j]
            left_node_type=nodeDoc.type[j] # e.g., PROPERTY, etc.     
            left_sm_id=nodeDoc.segmentId[j]
            
            if left_sm_id!=-1:
                    
                    for l in range (len(nodeDoc.mention)):
                            if (nodeDoc.segmentId[l]==left_sm_id):
                                
                                left_node_mn=nodeDoc.mention[l]
                                
           
                                
                                
            
            wBfrAft=computeWordsBfrAft(nodeDoc,j,tokenDoc)
            wordsBeforeLft=wBfrAft[0]
            
            for z in range(len(nodeDoc.mention)): # iterate over all nodes - to create the right side
                right_node_mn=nodeDoc.mention[z]
                right_node_type=nodeDoc.type[z]
                right_sm_id=nodeDoc.segmentId[z]
                
                
                if right_sm_id!=-1:
                            
                    for l in range (len(nodeDoc.mention)):
                            if (nodeDoc.segmentId[l]==right_sm_id):
                                right_node_mn=nodeDoc.mention[l]
                
                wBfrAft=computeWordsBfrAft(nodeDoc,z,tokenDoc)
                wordsAfterRgt=wBfrAft[1]
                
                
                
                
                if (j!=z and z!=rootIndex):
                    
                    wordsBetween=computeWordsBetween(nodeDoc,j,z,tokenDoc)
                    
                    segment_dif=str(z-j)
                    nOfWordsBetween=str(len(wordsBetween.split()))
                    
                    part_of=0
                    for k in range(len(goldDoc.left_mention)):
                        
                        
                        if j == goldDoc.left_id[k] and z == goldDoc.right_id[k]:
                            if goldDoc.type[k]=="equivalent":
                                part_of=2
                            elif goldDoc.type[k]=="part-of":
                                part_of=1
                                
                            
                            total_edges+=1
                            edges+=1
                            
                    
                    line=str(part_of)+"\t"+feat.process(left_node_mn,right_node_mn,left_node_type,right_node_type,wordsBetween,segment_dif,nOfWordsBetween,wordsBeforeLft,wordsAfterRgt)
                    writeList.append(line+"\n")
                    
        if (edges != len(nodeDoc.mention) - 1):
            print "invalid number of active relations: " + str(edges)+ " should be " + str(len(nodeDoc.mention) - 1)
            print "doc: " + nodeDoc.docId
    print ("")
    writeListToFile(outfile,writeList)


def createTrainVocabulary(feature_docs):   
    # create the train vocabulary sparse representation given the documents
    ds_string_list=[]
    
    
    labels=[]
    print "Create train sparse feature vectors..."
    for doc in feature_docs:  
        print ".",
        for idx in range(len(doc.lines)):
            
            line=doc.lines[idx] 
            label=doc.label[idx]
            ds_string_list.append(line)
            labels.append(label)

    vectorizer=CountVectorizer(token_pattern=r'\.*\S+',ngram_range=(1, 1))
    train_vec=vectorizer.fit_transform(ds_string_list)
    print ""
    print "Number of features " +str(len(vectorizer.get_feature_names()))
    
    return vectorizer,train_vec,labels

    
    
def selectIndicesFromList(mylist,idx_set):
    return  [mylist[i] for i in idx_set]

    
def splitTrainTest(feat_docs,node_docs,gold_docs,randomSeed):
    # split train and test set
    indices = np.arange(len(feat_docs))
    feat_docs_train, feat_docs_test, idx_train, idx_test = train_test_split(feat_docs, indices, test_size=0.15,random_state=randomSeed)
    node_docs_train=selectIndicesFromList(node_docs,idx_train)
    node_docs_test=selectIndicesFromList(node_docs,idx_test)
    gold_docs_train=selectIndicesFromList(gold_docs,idx_train)
    gold_docs_test=selectIndicesFromList(gold_docs,idx_test)
    
    return feat_docs_train, feat_docs_test, node_docs_train, node_docs_test, gold_docs_train, gold_docs_test    

def createTestVocabulary(feature_docs,vectorizer):   
    # create the test vocabulary sparse representation given the documents and the vectorizer
    ds_string_list=[]
    
    
    labels=[]
    print "Create test sparse feature vectors..."
    for doc in feature_docs:  
        print ".",
        for idx in range(len(doc.lines)):
            
            line=doc.lines[idx] 
            label=doc.label[idx]
            ds_string_list.append(line)
            labels.append(label)

    
    test_vec=vectorizer.transform(ds_string_list)
    print ""
    return test_vec,labels   
    
def classify(X_train,y_train,c,algorithm): # gives as output the classifier with making predictions

    if algorithm=="lr":
                clf=clf=LogisticRegression(C=c)         
    elif algorithm=="svm":
                clf=svm.LinearSVC(random_state=42,C=c)                                  
                                        
    clf.fit(X_train, y_train) 
    
    return clf

def classification(X_train,y_train,X_test,c,algorithm):# gives as output the classifier and the predictions
    if algorithm=="lr":
        clf=LogisticRegression(random_state=42,C=c)
    elif algorithm=="svm":
        clf=svm.LinearSVC(random_state=42,C=c)
        
    clf.fit(X_train, y_train) 
    y_pred=clf.predict(X_test)
    return clf,y_pred
    
class constraints: # execute permissible actions based on the predictions of the classifier for the greedy ransition based system
    def __init__(self,graphObject,clf,vectorizer):
        
        mch=machine()
        
        mch.stack.append(graphObject[0])
        mch.bufferL=graphObject[1:]
        
        
        self.stacks=[]
        self.buffers=[]
        self.actions=[]


        self.stacks.append(list(mch.stack))
        self.buffers.append(list(mch.bufferL))
        
        predsOverall=0
        
        while (mch.isNotInEndState()):
            

            preds=getPrediction(mch,clf,vectorizer,graphObject)[0]
            
            rankedActions,predsR=getRankedActions(clf,preds)

            stop=False
            cl=0
            while (stop==False):
                rAction=rankedActions[cl]
                
                if (mch.isPermissible(rAction)):
                    
                    predsOverall+=predsR[cl]
                    
                    if rAction=="left":
                        
                        mch.leftArc()
                    elif rAction=="right":
                        mch.rightArc()
                       
                    elif rAction=="swap":
                        mch.swap()
                    elif rAction=="shift":
                        mch.shift()

                    stop=True
                    
                    self.actions.append(rAction)
                    self.stacks.append(list(mch.stack))
                    self.buffers.append(list(mch.bufferL))
                    
                    
                else:
                    cl+=1
                   
        self.arcs=mch.arcs
        
        
        
def getLabelsFromPredictions(node_docs_test,clf,vectorizer): # predict the labels based on permissible actions on the transition based system
    
    pred_labels=[]
    
    print "Computing predictions..."
    for doc in node_docs_test:
        graphObj=[]
        print ".",
        
        for idx in range(len(doc.nodeId)):
            
            graphObj.append(graphObject(doc.mention[idx],doc.nodeId[idx],-1,doc.nodeId[idx]+"/"+doc.mention[idx],doc.type[idx]))

        con=constraints(graphObj,clf,vectorizer)
        predictions=con.arcs
        
        
        for lidx in range(len(doc.nodeId)):
            for ridx in range(len(doc.nodeId)):
                lnodeId=int(doc.nodeId[lidx])
                rnodeId=int(doc.nodeId[ridx])
                if lnodeId!=rnodeId and rnodeId!=0:

                    label=0
                    for arc in predictions:
                        if lnodeId==arc.left.nodeId and rnodeId==arc.right.nodeId:
                            label=1

                    pred_labels.append(label)
       
    print ""
    return pred_labels    
    
def getLabelsFromGoldFile(node_docs_test,goldFileDocsTest): # get the labels from the gold file 
    
    labels_gold=[]
    
    print "Get the test labels.."
    for doc in node_docs_test:
        graph=nx.DiGraph()
        
        print ".",
        for lidx in range(len(doc.nodeId)):
            for ridx in range(len(doc.nodeId)):
                lnodeId=int(doc.nodeId[lidx])
                rnodeId=int(doc.nodeId[ridx])
                
                lnodeMn=doc.mention[lidx]
                rnodeMn=doc.mention[ridx]

                if lnodeId!=rnodeId and rnodeId!=0:

                    label=0
                    
                    
                    for gf in goldFileDocsTest:
                        if doc.incrementalId==gf.incrementalId:
                            
                            for relIdx in range(len(gf.left_mention)):

                                    

                                    if lnodeId==int(gf.left_id[relIdx]) and rnodeId==int(gf.right_id[relIdx]):

                                        label=1
                                        graph.add_edge(str(lnodeId)+"/"+lnodeMn,str(rnodeId)+"/"+rnodeMn)

                            
                            labels_gold.append(label)
        
    print ""
    return labels_gold     
    
def getMTTLabels(X_train,X_test,node_docs_train,node_docs_test,y_train,c): # run the MTT pipeline

            w=np.zeros((1,X_train.shape[1]))

            labels_sparse = scipy.sparse.csr_matrix(map(int, y_train)) 
                            
            featuresActivated=labels_sparse.T.multiply(X_train)
                            
            featuresSum=scipy.sparse.csr_matrix.sum(featuresActivated,axis=0)
                                                       
            myargs = (X_train,node_docs_train)

            x,f,d=lbfgs.fmin_l_bfgs_b(mtt.L,x0=w,fprime=mtt.gradL,args=(X_train,node_docs_train,y_train,c,featuresSum),iprint=1,maxiter=1000)
                
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

def getLastToken(sequence):
    
    return ' '.join(sequence.split(' ')[-1:])
    
def getUnigramsToString(tag,term,endTag=""):
    unigrams_term_str=""
    unigrams_term=[]
    
    try:
        unigrams_vec=CountVectorizer(ngram_range=(1,1))
        unigrams_vec.fit_transform([term])
        unigrams_term=unigrams_vec.get_feature_names()
    except ValueError, e:
        pass
        #print 'Empty vocabulary : ' + " --ERROR "+  str(e)
                
    for i in range(len(unigrams_term)):
            unigrams_term_str+=tag+unigrams_term[i]+endTag+" "
    
    return unigrams_term_str

class tokenDoc:
    def __init__(self,id,iid):
        self.incrementalId=iid
        self.docId=id
        self.start=[]
        self.end=[]
        self.sf=[]
       
        
    def append(self,start,end,mention):
        self.start.append(start)
        self.end.append(end)        
        self.sf.append(mention)
        
class tokenParser:
    def __init__(self,file):
            docNr=-1
            self.token_docs=[]
            tokens=tokenDoc("","")      
            
            for i in range (file.shape[0]):
                if '#doc' in file[i][0] or i==file.shape[0]-1:#append all docs including the last one  
                    if (i==file.shape[0]-1):# append last line
                        tokens.append(file[i][0],file[i][1],file[i][2])
                    if (docNr!=-1):
                        self.token_docs.append(tokens)
                    docNr+=1
                    tokens=tokenDoc(file[i][0],docNr)                    
                else:
                    tokens.append(file[i][0],file[i][1],file[i][2])#append lines
                    
def splitbyDel(string,delimeter):                    
    
       return [x.strip() for x in string.split(delimeter)]
    

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results
    
    
def computeWordsBfrAft(crfDoc,row,tokenDoc): # get words before and after entities  
     
    crf_start=crfDoc.start[row]     
    crf_end=crfDoc.end[row]    
    wordsBefore=" "
    wordsAfter=" "
    if (crf_start!=0 and crf_end!=0):
                start_pos=-1
                end_pos=-1
                        
                for token in range (len(tokenDoc.sf)):
                                                        
                            if (int(tokenDoc.start[token])==int(crf_start) and start_pos==-1 ):
                                
                                start_pos=token
                                
                                
                            if (int(tokenDoc.end[token])==int(crf_end) and start_pos!=-1 and end_pos==-1):
                                
                                end_pos=token
                               
                                
                                
                                for i in range (start_pos-1,start_pos-3,-1):
                                        #print i
                                        if (i>=0):
                                            wordsBefore+=" "+tokenDoc.sf[i]  
                                
                                for i in range (end_pos+1,end_pos+3):
                                        if (i<=len(tokenDoc.sf)-1):
                                            wordsAfter+=" "+tokenDoc.sf[i]  
                                        
                                        
                       
    return wordsBefore,wordsAfter

def computeWordsBetween(crfDoc,l_row,r_row,tokenDoc): # get words between entities on the graph
    
    left_crf_start=crfDoc.start[l_row]     
    left_crf_end=crfDoc.end[l_row]        
    right_crf_start=crfDoc.start[r_row]
    right_crf_end=crfDoc.end[r_row]
    wordsBetween=" "
    if (left_crf_end!=0 and right_crf_end!=0):
                        
                        for token in range (len(tokenDoc.sf)):
                            
                            if (int(tokenDoc.end[token])>int(left_crf_end) and int(tokenDoc.start[token])<int(right_crf_start) ):
                                 wordsBetween+=" "+tokenDoc.sf[token]
                            if (int(tokenDoc.end[token])>int(right_crf_end) and int(tokenDoc.start[token])<int(left_crf_start) ):
                                 wordsBetween+=" "+tokenDoc.sf[token]
                       
    return wordsBetween



    
def replaceWhiteSpaces(string):
    return re.sub(r"\s+", '_', string)

class featurePreprocessing: # feature preprocessing on the graph based approach
    def __init__(self,feature_file):
        self.parent_f=read_properties(feature_file).getProperty("parent_feature")   
        self.child_f=read_properties(feature_file).getProperty("child_feature")   
        self.parent_child_concat=read_properties(feature_file).getProperty("parent_child_concat")   
        self.parent_type=read_properties(feature_file).getProperty("parent_type")   
        self.child_type=read_properties(feature_file).getProperty("child_type")   
        self.parent_child_type_concat=read_properties(feature_file).getProperty("parent_child_type_concat") 
        self.noun_feature=read_properties(feature_file).getProperty("noun_feature")   
        self.words_btw_concat=read_properties(feature_file).getProperty("words_btw_concat")   
        self.words_btw_unigrams=read_properties(feature_file).getProperty("words_btw_unigrams")   
        self.parent_unigrams=read_properties(feature_file).getProperty("parent_unigrams")
        self.child_unigrams=read_properties(feature_file).getProperty("child_unigrams")   
        self.num_of_words_btw=read_properties(feature_file).getProperty("num_of_words_btw")   
        self.segment_distance=read_properties(feature_file).getProperty("segment_distance")   
        self.segment_direction=read_properties(feature_file).getProperty("segment_direction")   
        self.wordsBeforeLft=read_properties(feature_file).getProperty("wordsBeforeLft")   
        self.wordsAfterRgt=read_properties(feature_file).getProperty("wordsAfterRgt")   
        
             
    def process(self,parent,child,parentType,childType,wordsBetween,segment_dif,nOfWordsBetween,words_before_lft,words_after_rgt):
       
    
        features=""
       
        PARENT_TAG="parent#|"
        CHILD_TAG="child#|"
        PARENT_TYPE_TAG="parentType#|"
        CHILD_TYPE_TAG="childType#|"
        PARENT_CHILD_CONCAT_TAG=PARENT_TAG+ CHILD_TAG
        
        NOUN_FEATURE=PARENT_CHILD_CONCAT_TAG+"noun#|"
        PATTERN_TAG="pattern#|"
        SEGMENT_DIF_TAG="segment_dif#|"
        SEGMENT_DIRECTION_TAG="segment_dir#|"
        NWORDS_BTW_TAG="numOfWordsBtw#|"
        WORDS_BEFORE_PARENT_TAG="words_bfr_par#|"
        WORDS_AFTER_CHILD_TAG="words_aft_chi#|"
        
        
        if self.parent_f=="True":
            feature_set_1=PARENT_TAG+replaceWhiteSpaces(parent)#parent
            features+=feature_set_1
        
        features+=" "
        
        if self.child_f=="True":
            feature_set_2=CHILD_TAG+replaceWhiteSpaces(child)#child
            features+=feature_set_2       
        
        features+=" "
        
        if self.parent_child_concat=="True":
            feature_set_3=PARENT_CHILD_CONCAT_TAG+replaceWhiteSpaces(parent)+replaceWhiteSpaces(child)#parent #child concatenation
            features+=feature_set_3
            
        features+=" "
            
        if self.parent_type =="True":
            feature_type_set_1=PARENT_TYPE_TAG+replaceWhiteSpaces(parentType)
            features+=feature_type_set_1
            
        features+=" "   
            
        if self.child_type =="True":
            feature_type_set_2=CHILD_TYPE_TAG+replaceWhiteSpaces(childType)
            features+=feature_type_set_2          
            
        features+=" "

        if self.parent_child_type_concat=="True":
            
            feature_type_set_3=PARENT_TYPE_TAG+ CHILD_TYPE_TAG+replaceWhiteSpaces(parentType)+replaceWhiteSpaces(childType)
            features+=feature_type_set_3

        features+=" "
            
            
        if self.noun_feature=="True":
            feature_set_4=NOUN_FEATURE+getLastToken(parent)+getLastToken(child)    
            features+=feature_set_4
            
        #feature_set_5=(PARENT_TAG+ floorNormalization(parent.encode("utf8")).replace(" ","_")).decode("utf8")
        #feature_set_6=(CHILD_TAG+ floorNormalization(child.encode("utf8")).replace(" ","_")).decode("utf8")
        features+=" "
        

        if self.words_btw_concat=="True":
            feature_set_wordsBetween=PATTERN_TAG+PARENT_TAG+replaceWhiteSpaces(wordsBetween)+CHILD_TAG
            features+=feature_set_wordsBetween

        features+=" "

        if self.words_btw_unigrams=="True":
            unigrams_words_between=getUnigramsToString(PATTERN_TAG+PARENT_TAG,wordsBetween,CHILD_TAG)            
            features+=unigrams_words_between

        features+=" "

        if self.parent_unigrams=="True":
            unigrams_parent_str=getUnigramsToString(PARENT_TAG,parent)
            features+=unigrams_parent_str
            
        features+=" "
            
        if self.child_unigrams=="True":
            unigrams_child_str=getUnigramsToString(CHILD_TAG,child)  
            features+=unigrams_child_str
            
        features+=" "
        
            
        if self.num_of_words_btw=="True":
                    
            feature_nwordsbtwn=NWORDS_BTW_TAG+nOfWordsBetween
            features+=feature_nwordsbtwn    

        features+=" "            
                
        if self.segment_distance=="True":
            
            feature_segment_dif=SEGMENT_DIF_TAG+segment_dif
            features+=feature_segment_dif
            
        features+=" "  
        
        if self.segment_direction=="True":
            dir=-1
            if int(segment_dif)>0:
                dir=1
            else:
                dir=2
            
            feature_segment_dir=SEGMENT_DIRECTION_TAG+str(dir)
            features+=feature_segment_dir
            
        features+=" "  
        
      

        if self.wordsBeforeLft=="True":
            
            feature_words_before_lft=WORDS_BEFORE_PARENT_TAG+replaceWhiteSpaces(words_before_lft)
            features+=feature_words_before_lft
            
        features+=" "    

        if self.wordsAfterRgt=="True":
            
            feature_words_after_rgt=WORDS_AFTER_CHILD_TAG+replaceWhiteSpaces(words_after_rgt)
            features+=feature_words_after_rgt
            
        features+=" "       
        
       
        return features  

def queryRelation(parent,child,parentType,childType,vectorizer,clf):
    return clf.decision_function(vectorizer.transform([featurePreprocessing(parent,child,parentType,childType)]))  

def queryFeature(feature,clf,vectorizer):
    return clf.decision_function(vectorizer.transform([feature]))  
    
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
            #left_node_type=nodeDoc.type[j]       
            
           
            for z in range(len(nodeDoc.mention)):
                right_node_mn=nodeDoc.mention[z]
                #right_node_type=nodeDoc.type[z]
                rel=""
                
                if (j!=z and z!=rootIndex):
                    label=0
                    for ed in range (len(graphDoc.graph.edges())):
                
                            left=graphDoc.graph.edges()[ed][0].split("/")
                            right=graphDoc.graph.edges()[ed][1].split("/")  
                            
                            left_id=int(left[0])
                            right_id=int (right[0])
                            #print left_id
                            #print right_id
                            if (j==left_id and z==right_id):
                                      label=1                          
                                
                    ptr+=1               
                    y_edmonds_train.append(label)
                    
    print ""        
    return y_edmonds_train

def draw_func_thres(G):
    
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0]
    pos=graphviz_layout(G,with_labels=True) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,pos)

    # edges
    nx.draw_networkx_edges(G,pos,edgelist=elarge)

    nx.draw_networkx_labels(G,pos)
def delFileIfExists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    
def draw_func_all_windows(G,path="",save=False):
    pos = graphviz_layout(G)
    
    nx.draw(G, pos)
   
    nx.draw_networkx_labels(G,pos,node_size=2000,node_shape='o',node_color='0.75')
    if save==True:
         delFileIfExists(path)
     
         plt.savefig(path)
         plt.clf()
         plt.close()


def draw_func_all_linux(G):
	pos = graphviz_layout(G)
	nx.draw(G, pos)
	nx.draw_networkx_labels(G,pos)


def computeDocIDs(file):
    doc_ids=[]
    for i in range (file.shape[0]):
        if '#doc' in file[i][0]:
                doc_ids.append(file[i][0])
    return doc_ids

def getDocID(ids,id):
    return [i for i,x in enumerate(ids) if id in x]

def getSeqIDFromDocID(ids,id):
    return ids[id]


class goldFileDoc:
    def __init__(self,id,iid):
        self.incrementalId=iid
        self.docId=id
        self.form=[]
        self.left_id=[]
        self.right_id=[]
        self.type=[]
        self.left_mention=[]
        self.right_mention=[]
        
    def append(self,form,left_id,right_id,type,left_mention,right_mention):
        self.form.append(form)
        self.left_id.append(int(left_id))
        self.right_id.append(int(right_id))
        self.type.append(type)
        self.left_mention.append(left_mention)
        self.right_mention.append(right_mention)
        

class goldFileParser:
    def __init__(self,file):
            docNr=-1
            self.gold_docs=[]
            featureDoc=goldFileDoc("","")      
            
            for i in range (file.shape[0]):
                if '#doc' in file[i][0] or i==file.shape[0]-1: 
                    if (i==file.shape[0]-1):
                        featureDoc.append(file[i][0],file[i][1],file[i][2],file[i][3],file[i][4],file[i][5])
                    if (docNr!=-1):
                        self.gold_docs.append(featureDoc)
                    docNr+=1
                    featureDoc=goldFileDoc(file[i][0],docNr)                    
                else:
                    featureDoc.append(file[i][0],file[i][1],file[i][2],file[i][3],file[i][4],file[i][5])
                    


                    
class graphDoc:
    def __init__(self,id,iid,graph):
        self.incrementalId=iid
        self.docId=id
        self.graph=graph
        
        
        
class featuresFileParser:
    def __init__(self,file):
            docNr=-1
            #print file
            self.feature_docs=[]
            featureDoc=featureFileDoc("","")      
            
            for i in range (file.shape[0]):
                if '#doc' in file[i][0] or i==file.shape[0]-1: 
                    if (i==file.shape[0]-1):
                        left_mn_rel=find_between(file[i][1],"parent#|"," child#|").replace("_"," ")
                        #print left_mn_rel
                        right_mn_rel=find_between(file[i][1],"child#|"," ").replace("_"," ")
                        #print right_mn_rel
                        label=file[i][0]
                        featureDoc.append(left_mn_rel,right_mn_rel,label,file[i][1])
                    if (docNr!=-1):
                        self.feature_docs.append(featureDoc)
                    docNr+=1
                    featureDoc=featureFileDoc(file[i][0],docNr)                    
                else:
                    left_mn_rel=find_between(file[i][1],"parent#|"," child#|").replace("_"," ")
                    right_mn_rel=find_between(file[i][1],"child#|"," ").replace("_"," ")
                    label=file[i][0]
                    featureDoc.append(left_mn_rel,right_mn_rel,label,file[i][1])
class featureFileDoc:
    def __init__(self,id,iid):
        self.incrementalId=iid
        self.docId=id
        self.label=[]
        self.left_mention=[]
        self.right_mention=[]
        self.lines=[]
        
    def append(self,left_mention,right_mention,label,line):
        self.label.append(label)
        self.left_mention.append(left_mention)
        self.right_mention.append(right_mention)
        self.lines.append(line)


    
    

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


class nodesDoc:
    def __init__(self,id,iid):
        self.incrementalId=iid
        self.docId=id
        self.start=[]
        self.end=[]
        self.type=[]
        self.mention=[]
        self.position=[]
        self.segmentId=[]
        self.nodeId=[]
        
    def append(self,start,end,type,mention,nodeId,segmentId):
        self.start.append(int(start))
        self.end.append(int(end))
        self.type.append(type)
        self.mention.append(mention)
        #self.position.append(position)
        self.nodeId.append(nodeId)
        self.segmentId.append(int(segmentId))
        

        
class nodeParser:
    def __init__(self,file):
            docNr=-1
            self.node_docs=[]
            node=nodesDoc("","")      
            
            for i in range (file.shape[0]):
                if '#doc' in file[i][0] or i==file.shape[0]-1:#append all docs including the last one  
                    if (i==file.shape[0]-1):# append last line
                        segmentId=file[i][5].split(" ")[1]
                        nodeId=file[i][4].split(" ")[1]
                        node.append(file[i][0],file[i][1],file[i][2],file[i][3],nodeId,segmentId)
                    if (docNr!=-1):
                        self.node_docs.append(node)
                    docNr+=1
                    node=nodesDoc(file[i][0],docNr)                    
                else:
                    segmentId=file[i][5].split(" ")[1]
                    nodeId=file[i][4].split(" ")[1]
                    node.append(file[i][0],file[i][1],file[i][2],file[i][3],nodeId,segmentId)#append lines

                    
class read_properties:
	def __init__(self,filepath, sep='=', comment_char='#'):
	        """Read the file passed as parameter as a properties file."""
		self.props = {}
		#print filepath
		with open(filepath, "rt") as f:
			for line in f:
			    #print line
			    l = line.strip()
			    if l and not l.startswith(comment_char):
                		key_value = l.split(sep)
                		self.props[key_value[0].strip()] = key_value[1].strip('" \t') 
		
	def getProperty(self,propertyName):
		return self.props.get(propertyName)

         

def  getRankedActions(clf,preds):

    rankActions=[]
    predsR=[]
    sortedPreds=sorted(preds)
    for spred in reversed(sortedPreds):
        
        for i in range(len(preds)):
            pred=preds[i]
            if pred==spred:
                rankActions.append(clf.classes_[i])
                predsR.append(pred)
            
    return rankActions,predsR
    
def getPrediction(machine,clf,vectorizer,graphObj):
    stack=machine.stack
    bufferL=machine.bufferL
    arcs=machine.arcs
    
    configFeatures=featurePreprocessingTransitions().process(stack,bufferL,arcs,graphObj)
    
    testSparse=vectorizer.transform([configFeatures])
    
    return clf.decision_function(testSparse)
    
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


def printResultsToFile(y_test,y_pred,filename,c,model,random_seed):

            binary_confusion_matrix = BinaryConfusionMatrix(map(int, y_test), y_pred)

            tp=float(binary_confusion_matrix.TP)
            tn=binary_confusion_matrix.TN
            fp=binary_confusion_matrix.FP
            fn=binary_confusion_matrix.FN

             
            f1=(2*tp)/(2*tp+fp+fn)                                                       
            re=tp / (tp + fn)
            pre=tp / (tp + fp)
            
            write_list=[]

            write_list.append("{0:15s}   {1:12s}   {2:10s}".format("C","model","seed")+"\n")
            write_list.append("{0:15s}   {1:12s}   {2:10s}".format(str(int(c)),model,str(random_seed))+"\n")

            write_list.append("{0:15s}   {1:12s}   {2:12s}   {3:12s}   {4:12s}   {5:12s}   {6:12s}".format( "Type", "TP", "FP", "FN", "Pr", "Re", "F1")+"\n")
            write_list.append(("{0:15s}   {1:12s}   {2:12s}   {3:12s}   {4:12s}   {5:12s}   {6:12s}".format("part-of",str(int(tp)),str(fp),str(fn),str(round(pre,4)),str(round(re,4)),str(round(f1,4)))+"\n")+"\n")
                        
            appendListToFile(filename,write_list)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
