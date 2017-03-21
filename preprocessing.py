import numpy as np
import networkx as nx
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import transition_utils,graph_utils,utilities,file_parsers


# features used in the transition based parser        
class transitionBasedFeaturePreprocessing:
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
         
        
        sortedSuccessors=transition_utils.sortNodeList(successors)
        

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

        
        graph=transition_utils.graphFromArcObject(arcs)
        
        
        leftW,rightW,leftT,rightT,sortedSuccessors=self.getLeftRightMost(arcs,graph,stack[len(stack)-1],graphObject)  
        
       
        arc0=""
        for arcIdx in range(len(arcs)):
            
            if arcIdx==len(arcs)-1:
                arc0=arcs[arcIdx].left.name + "_" +arcs[arcIdx].right.name + "_" +arcs[arcIdx].transition
        
        
        if self.s0_type==True:
            feature_set_1=s0_typeTAG+utilities.replaceWhiteSpaces(stack[len(stack)-1].type)#s0_type
            features+=feature_set_1
        
        features+=" "
        
        
        if self.s1_type==True and len(stack)>1:
            feature_set_2=s1_typeTAG+utilities.replaceWhiteSpaces(stack[len(stack)-2].type)#s1_type
            features+=feature_set_2
        
        features+=" "
        
        
        if self.s0_word==True:
            feature_set_3=s0_wordTAG+utilities.replaceWhiteSpaces(stack[len(stack)-1].name)#s0_word
            features+=feature_set_3
        
        features+=" "
        
        
        if self.s1_word==True and len(stack)>1:
            feature_set_4=s1_wordTAG+utilities.replaceWhiteSpaces(stack[len(stack)-2].name)#s1_word
            features+=feature_set_4
        
        features+=" "
        
       
        
        if self.b0_type==True and len(bufferL)>0:
            feature_set_5=b0_typeTAG+utilities.replaceWhiteSpaces( bufferL[0].type)#b0_type
            features+=feature_set_5
        
        features+=" "
        
        if self.b0_word==True and len(bufferL)>0:
            feature_set_6=b0_wordTAG+utilities.replaceWhiteSpaces( bufferL[0].name)#b0_word
            features+=feature_set_6
        
        features+=" "
        
        
        if self.b1_type==True and len(bufferL)>1:
            feature_set_7=b1_typeTAG+utilities.replaceWhiteSpaces( bufferL[1].type)#b1_type
            features+=feature_set_7
        
        features+=" "
        
        if self.b1_word==True and len(bufferL)>1:
            feature_set_8=b1_wordTAG+utilities.replaceWhiteSpaces( bufferL[1].name)#b1_word
            features+=feature_set_8
        
        features+=" "
        
        
        if self.b2_type==True and len(bufferL)>2:
            feature_set_9=b2_typeTAG+utilities.replaceWhiteSpaces( bufferL[2].type)#b1_type
            features+=feature_set_9
        
        features+=" "
        
        if self.b2_word==True and len(bufferL)>2:
            feature_set_10=b2_wordTAG+utilities.replaceWhiteSpaces( bufferL[2].name)#b1_word
            features+=feature_set_10
        
        features+=" "
        
        if self.b3_type==True and len(bufferL)>3:
            feature_set_11=b3_typeTAG+utilities.replaceWhiteSpaces( bufferL[3].type)#b1_type
            features+=feature_set_11
        
        features+=" "
        
        if self.b3_word==True and len(bufferL)>3:
            feature_set_12=b3_wordTAG+utilities.replaceWhiteSpaces( bufferL[3].name)#b1_word
            features+=feature_set_12
        
        features+=" "
        
        
        if self.a0_word==True:
            feature_set_13=a0_wordTAG+utilities.replaceWhiteSpaces( arc0)#a0_word
            features+=feature_set_13
        
        features+=" "
        
        if self.left_s0_word==True:
            feature_set_14=leftMost_s0_word_TAG+utilities.replaceWhiteSpaces(leftW)#a0_word
            features+=feature_set_14
        
        features+=" "
        
        if self.right_s0_word==True:
            feature_set_15=rightMost_s0_word_TAG+utilities.replaceWhiteSpaces(rightW)#a0_word
            features+=feature_set_15
        
        features+=" "
        
        
        if self.s0_successors==True:
        
            for node in successors:
                node=''.join(node.split("/")[1:])
                features+=s0_successors_TAG+node
                features+=" "
        
        features+=" "
        
        
        if self.s0_b0_word==True :
            feature_set_16=s0_b0_word_TAG+utilities.replaceWhiteSpaces(stack[len(stack)-1].name)#a0_word
            
            if len(bufferL)>0:
            
                feature_set_16+="_"+utilities.replaceWhiteSpaces(bufferL[0].name)
            features+=feature_set_16
        
        features+=" "
        
        
        if self.s0_s1_word==True :
            feature_set_17=s0_s1_word_TAG+utilities.replaceWhiteSpaces(stack[len(stack)-1].name)#a0_word
            
            if len(stack)>1:
            
                feature_set_17+="_"+utilities.replaceWhiteSpaces(stack[len(stack)-2].name)
            features+=feature_set_17
        
        features+=" "
        
        
        if self.b0_b1_word==True :
            feature_set_18=b0_b1_word_TAG
            if len(bufferL)>0:
                feature_set_18+=utilities.replaceWhiteSpaces(bufferL[0].name)#a0_word
            
            if len(bufferL)>1:
            
                feature_set_18+="_"+utilities.replaceWhiteSpaces(bufferL[1].name)
            features+=feature_set_18
        
        features+=" "
        
        
        
        if self.s0_b0_type==True :
            feature_set_19=s0_b0_type_TAG+utilities.replaceWhiteSpaces(stack[len(stack)-1].type)#a0_type
            
            if len(bufferL)>0:
            
                feature_set_19+="_"+utilities.replaceWhiteSpaces(bufferL[0].type)
            features+=feature_set_19
        
        features+=" "
        
        
        if self.s0_s1_type==True :
            feature_set_20=s0_s1_type_TAG+utilities.replaceWhiteSpaces(stack[len(stack)-1].type)#a0_word
            
            if len(stack)>1:
            
                feature_set_20+="_"+utilities.replaceWhiteSpaces(stack[len(stack)-2].type)
            features+=feature_set_20
        
        features+=" "
        
        
        if self.b0_b1_type==True :
            feature_set_21=b0_b1_type_TAG
            if len(bufferL)>0:
                feature_set_21+=utilities.replaceWhiteSpaces(bufferL[0].type)#a0_word
            
            if len(bufferL)>1:
            
                feature_set_21+="_"+utilities.replaceWhiteSpaces(bufferL[1].type)
            features+=feature_set_21
        
        features+=" "
        
        
        if self.left_s0_type==True:
            feature_set_22=leftMost_s0_type_TAG+utilities.replaceWhiteSpaces(leftT)#a0_word
            features+=feature_set_22
        
        features+=" "
        
        if self.right_s0_type==True:
            feature_set_23=rightMost_s0_type_TAG+utilities.replaceWhiteSpaces(rightT)#a0_word
            features+=feature_set_23
        
        features+=" "
        
        
        if self.s1_b0_word==True :
            feature_set_24=s1_b0_word_TAG
            if len(stack)>1:
                feature_set_24+=utilities.replaceWhiteSpaces(stack[len(stack)-2].name)#a0_word
            
            if len(bufferL)>0:
            
                feature_set_24+="_"+utilities.replaceWhiteSpaces(bufferL[0].name)
            features+=feature_set_24
        
        features+=" "
        
        
        if self.s1_b0_type==True :
            feature_set_26=s1_b0_type_TAG
            if len(stack)>1:
                feature_set_26+=utilities.replaceWhiteSpaces(stack[len(stack)-2].type)#a0_word
            
            if len(bufferL)>0:
            
                feature_set_26+="_"+utilities.replaceWhiteSpaces(bufferL[0].type)
            features+=feature_set_26
        
        features+=" "
        
        
        
        
        if self.s1_s0_b0_word==True :
            feature_set_25=s1_s0_b0_word_TAG
            if len(stack)>1:
                feature_set_25+="s1_"+utilities.replaceWhiteSpaces(stack[len(stack)-2].name)#a0_word
                
            if len(stack)>0:
                feature_set_25+="s0_"+utilities.replaceWhiteSpaces(stack[len(stack)-2].name)#a0_word
            
            if len(bufferL)>0:
            
                feature_set_25+="b0_"+utilities.replaceWhiteSpaces(bufferL[0].name)
            features+=feature_set_25
        
        features+=" "
        
        return features  



def createFeaturesFile(goldFile,nodeFile,outfile,tokenfile,featuresFile):

    # create graph based features given the relations 
    feat=graphBasedFeaturePreprocessing(featuresFile)
    total_edges=0
    total_segments=0
    
    gold_docs=file_parsers.goldFileParser(goldFile).gold_docs
    node_docs=file_parsers.nodeParser(nodeFile).node_docs
    
    token_docs=file_parsers.tokenParser(tokenfile).token_docs

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
    utilities.writeListToFile(outfile,writeList)





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
    
    
class graphBasedFeaturePreprocessing: # feature preprocessing on the graph based approach
    def __init__(self,feature_file):
        self.parent_f=file_parsers.read_properties(feature_file).getProperty("parent_feature")   
        self.child_f=file_parsers.read_properties(feature_file).getProperty("child_feature")   
        self.parent_child_concat=file_parsers.read_properties(feature_file).getProperty("parent_child_concat")   
        self.parent_type=file_parsers.read_properties(feature_file).getProperty("parent_type")   
        self.child_type=file_parsers.read_properties(feature_file).getProperty("child_type")   
        self.parent_child_type_concat=file_parsers.read_properties(feature_file).getProperty("parent_child_type_concat") 
        self.noun_feature=file_parsers.read_properties(feature_file).getProperty("noun_feature")   
        self.words_btw_concat=file_parsers.read_properties(feature_file).getProperty("words_btw_concat")   
        self.words_btw_unigrams=file_parsers.read_properties(feature_file).getProperty("words_btw_unigrams")   
        self.parent_unigrams=file_parsers.read_properties(feature_file).getProperty("parent_unigrams")
        self.child_unigrams=file_parsers.read_properties(feature_file).getProperty("child_unigrams")   
        self.num_of_words_btw=file_parsers.read_properties(feature_file).getProperty("num_of_words_btw")   
        self.segment_distance=file_parsers.read_properties(feature_file).getProperty("segment_distance")   
        self.segment_direction=file_parsers.read_properties(feature_file).getProperty("segment_direction")   
        self.wordsBeforeLft=file_parsers.read_properties(feature_file).getProperty("wordsBeforeLft")   
        self.wordsAfterRgt=file_parsers.read_properties(feature_file).getProperty("wordsAfterRgt")   
        
             
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
            feature_set_1=PARENT_TAG+utilities.replaceWhiteSpaces(parent)#parent
            features+=feature_set_1
        
        features+=" "
        
        if self.child_f=="True":
            feature_set_2=CHILD_TAG+utilities.replaceWhiteSpaces(child)#child
            features+=feature_set_2       
        
        features+=" "
        
        if self.parent_child_concat=="True":
            feature_set_3=PARENT_CHILD_CONCAT_TAG+utilities.replaceWhiteSpaces(parent)+utilities.replaceWhiteSpaces(child)#parent #child concatenation
            features+=feature_set_3
            
        features+=" "
            
        if self.parent_type =="True":
            feature_type_set_1=PARENT_TYPE_TAG+utilities.replaceWhiteSpaces(parentType)
            features+=feature_type_set_1
            
        features+=" "   
            
        if self.child_type =="True":
            feature_type_set_2=CHILD_TYPE_TAG+utilities.replaceWhiteSpaces(childType)
            features+=feature_type_set_2          
            
        features+=" "

        if self.parent_child_type_concat=="True":
            
            feature_type_set_3=PARENT_TYPE_TAG+ CHILD_TYPE_TAG+utilities.replaceWhiteSpaces(parentType)+utilities.replaceWhiteSpaces(childType)
            features+=feature_type_set_3

        features+=" "
            
            
        if self.noun_feature=="True":
            feature_set_4=NOUN_FEATURE+getLastToken(parent)+getLastToken(child)    
            features+=feature_set_4
            
        #feature_set_5=(PARENT_TAG+ floorNormalization(parent.encode("utf8")).replace(" ","_")).decode("utf8")
        #feature_set_6=(CHILD_TAG+ floorNormalization(child.encode("utf8")).replace(" ","_")).decode("utf8")
        features+=" "
        

        if self.words_btw_concat=="True":
            feature_set_wordsBetween=PATTERN_TAG+PARENT_TAG+utilities.replaceWhiteSpaces(wordsBetween)+CHILD_TAG
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
            
            feature_words_before_lft=WORDS_BEFORE_PARENT_TAG+utilities.replaceWhiteSpaces(words_before_lft)
            features+=feature_words_before_lft
            
        features+=" "    

        if self.wordsAfterRgt=="True":
            
            feature_words_after_rgt=WORDS_AFTER_CHILD_TAG+utilities.replaceWhiteSpaces(words_after_rgt)
            features+=feature_words_after_rgt
            
        features+=" "       
        
       
        return features  
        
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
    
    


def getLastToken(sequence):
    
    return ' '.join(sequence.split(' ')[-1:])
    
  
    
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