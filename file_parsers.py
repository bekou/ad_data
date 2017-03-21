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
                    
                    
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
