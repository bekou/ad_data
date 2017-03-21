from sklearn.linear_model import LogisticRegression
from sklearn import svm
from pandas_confusion import ConfusionMatrix,BinaryConfusionMatrix
import os,errno,re,codecs,unidecode

#functions for fixing unicode issues  
def remove_non_ascii(text):
    return unidecode(unicode(text, encoding = "utf-8"))
def fixUnicode(text):
      return str(text).decode('ascii', 'ignore')
      
def writeListToFile(filename,lst):
    with codecs.open(filename, 'w',"utf-8") as f:
        for s in lst:
            
            f.write(s )

def appendListToFile(filename,lst):
            
    with codecs.open(filename, 'a',"utf-8") as f:
        for s in lst:
            
            f.write(s )
            
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
    

def splitbyDel(string,delimeter):                    
    
       return [x.strip() for x in string.split(delimeter)]
    

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results
    
def replaceWhiteSpaces(string):
    return re.sub(r"\s+", '_', string)
    
       
def delFileIfExists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
 
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

            
        
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