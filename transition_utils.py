import networkx as nx
import preprocessing

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
    
    configFeatures=preprocessing.transitionBasedFeaturePreprocessing().process(stack,bufferL,arcs,graphObj)
    
    testSparse=vectorizer.transform([configFeatures])
    
    return clf.decision_function(testSparse)

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