import graph_utils
import transition_utils
import file_parsers
import preprocessing
import utilities
import pandas as pd 
import os
import matrix_tree_theorem as mtt



class dependency_parsing:

        def __init__(self,config_file,graphFeaturesFile,model):
                
                
                self.features_numbers=""
                self.config_file=config_file                
                self.model=model
                self.features=graphFeaturesFile # features to use
                
                
                #############################################################################################################
                ###Check that directories exist in the filesystem############################################################
                #############################################################################################################
                self.results_dir=file_parsers.read_properties(config_file).getProperty("base_system_dir")+"results"#results directory
                utilities.make_sure_path_exists(self.results_dir)
                self.data_dir=file_parsers.read_properties(config_file).getProperty("base_data_dir")#data directory
                utilities.make_sure_path_exists(self.data_dir)
                
                #############################################################################################################
                ###Input files###############################################################################################
                #############################################################################################################
                
                #the tokens file for all the documents
                self.tokensFile=self.data_dir+file_parsers.read_properties(config_file).getProperty("tokensfile")
                self.tokens_file=pd.read_csv(self.tokensFile,encoding="utf-8",engine='python').as_matrix()

                
                #the relation file for all the documents (e.g., RELATION	2	0	part-of	ROOT    villa)
                if model=="transition":
                    self.goldFileName=self.data_dir+file_parsers.read_properties(config_file).getProperty("goldfile_tb")
                else:
                    self.goldFileName=self.data_dir+file_parsers.read_properties(config_file).getProperty("goldfile_gr")
                   
                
                #the node file for all the documents (e.g., 0	0	ROOT_TYPE	ROOT	NODE 2)
                if model=="transition":
                    self.nodeFile=file_parsers.read_properties(config_file).getProperty("node_file_tb")
                            
                else:
                    self.nodeFile=file_parsers.read_properties(config_file).getProperty("node_file_gr")
                self.nodeFilename=self.data_dir+self.nodeFile  
                
                # Read the node file (e.g., 0	0	ROOT_TYPE	ROOT	NODE 0	SEGMENT -1) - root's segment id is -1 since it is a virtual segment        
                self.node_col_vector=['start_char#','end_char#','type#','sf#','nodeId#','segmentId#']
                self.node_file=pd.read_csv(self.nodeFilename,names=self.node_col_vector,sep="\t",encoding="utf-8",engine='python').as_matrix()
                                
                
                
                #############################################################################################################
                ###Output files##############################################################################################
                #############################################################################################################
                
                #similar to goldFileName but the relations are guaranteed to be a tree
                if model=="transition":
                    self.goldTreeFile=file_parsers.read_properties(config_file).getProperty("gold_tree_file_tb")
                                   
                else:
                    self.goldTreeFile=file_parsers.read_properties(config_file).getProperty("gold_tree_file_gr")
                
                self.goldTreeFilename=self.data_dir+self.goldTreeFile   # gold file produced after running edmonds on the whole set to guarantee tree structure
                
                #the graph based output features file
                self.graph_features_file=file_parsers.read_properties(config_file).getProperty("graph_features_file")
                self.graphfeaturesOutput=self.data_dir+self.graph_features_file
                
                #the transition based output features file
                self.transition_file=file_parsers.read_properties(config_file).getProperty("transitions_features_file")
                self.transitionfeaturesOutput=self.data_dir+self.transition_file
                
                                    
                ###Read the relation file (e.g., RELATION	0	3	part-of	 appartement    hall)### 
                self.gold_vector=['arg','left_id','right_id','rel_type','left_mention','right_mention']
                gold_file=pd.read_csv(self.goldFileName,names=self.gold_vector,sep="\t",encoding="utf-8").as_matrix()
                
                ###Transform the relation file to a graph### 
                if  os.path.exists(self.goldTreeFilename)==False:
                    self.goldDocsGraph=graph_utils.GraphsFromGoldFile(gold_file)
                    ###Run the edmond algorithm on the input graph to guarantee that the input is ### 
                    self.goldSpanningDocsGraph=graph_utils.EdmondGraphs(self.goldDocsGraph)
                
                    graph_utils.writeDocsGraphsToFile(self.node_file,self.goldSpanningDocsGraph,self.goldTreeFilename)
                    
                
                # Load the tree relation file 
                gold_vector=['arg','left_id','right_id','rel_type','left_mention','right_mention']
                self.gold_tree_file=pd.read_csv(self.goldTreeFilename,names=gold_vector,sep="\t",encoding="utf-8",engine='python').as_matrix()
               
                graphs=graph_utils.GraphsFromGoldFile(self.gold_tree_file)
                
                
                self.node_docs=file_parsers.nodeParser(self.node_file).node_docs
                
                #############################################################################################################
                ###Create or load the features file if it is not already computed ###########################################
                #############################################################################################################
                
                    
                
                if (model=="mtt" or model=="edmond" or model=="threshold") and os.path.exists(self.graphfeaturesOutput)==False :      
                            
                            preprocessing.createFeaturesFile(self.gold_tree_file,self.node_file,self.graphfeaturesOutput,self.tokens_file,self.features)
                            
               
                                
                elif  model == "transition" and os.path.exists(self.transitionfeaturesOutput)==False :
                    
                    train_file=[]
                    for graph in graphs:
                                         
                                        train_file.append(graph.docId+"\n")
                                        
                                        
                                        traversal=transition_utils.inOrder(graph.graph,"0/ROOT",[])      
                                         
                                        graphObjects=transition_utils.parseGraphs(graph,traversal,self.node_docs)
                                        oracle=transition_utils.Oracle(graphObjects,graph)


                                        for i in range (len(oracle.stacks)-1):
                                            
                                            train_file.append(oracle.actions[i] + "\t"+preprocessing.transitionBasedFeaturePreprocessing().process(oracle.stacks[i],oracle.buffers[i],oracle.arcs[i],graphObjects) + "\n")
                                            
                    utilities.writeListToFile(self.transitionfeaturesOutput,train_file)
                    
                    
                    
                try:
                                
                            vec=['labels','features']
                            
                            if model=="mtt" or model=="edmond" or model=="threshold": # graph based features
                                  
                                self.features_file=pd.read_csv(self.graphfeaturesOutput,names=vec,sep="\t",encoding="utf-8",engine='python').as_matrix()   
                                
                            elif model == "transition": # transition-based features
                                self.features_file=pd.read_csv(self.transitionfeaturesOutput,names=vec,sep="\t",encoding="utf-8",engine='python').as_matrix() 
                                                                    
                            print ("Gold tree relation features file succesfully loaded\n")
                except ValueError, e:
                                print 'Consider to call computeFeatures : ' + " --ERROR "+  str(e)
 

        
        def computeScores(self,c,randomSeed):
        
            #read the relation file, the node file 
            gold_docs=file_parsers.goldFileParser(self.gold_tree_file).gold_docs
            node_docs=self.node_docs
            feat_docs=file_parsers.featuresFileParser(self.features_file).feature_docs
            
            # random split to train and test set 
            feat_docs_train, feat_docs_test, node_docs_train, node_docs_test, gold_docs_train, gold_docs_test=preprocessing.splitTrainTest(feat_docs,node_docs,gold_docs,randomSeed)
            
            # transform the documents to sparse representations
            train_voc=preprocessing.createTrainVocabulary(feat_docs_train)

            vectorizer=train_voc[0]#vectorizer

            X_train=train_voc[1].astype('double')#train features -vector - sparse 
            y_train=train_voc[2]#labels
            
            
            test_voc=preprocessing.createTestVocabulary(feat_docs_test,vectorizer)
 
                                
            X_test=test_voc[0]
            y_test=test_voc[1]  
            
            # run the various models
            if self.model=="mtt":
            
                pred_labels=graph_utils.getMTTLabels(X_train,X_test,node_docs_train,node_docs_test,y_train,c)
            
            

            if self.model=="threshold":    
            
                clf_dec=utilities.classification(X_train,y_train,X_test,c,"lr") #logistic regression - threshold
                clf=clf_dec[0]#classifier                
                pred_labels=clf_dec[1]#predictions
                pred_labels=map(int, pred_labels)
                
            if self.model=="edmond":    
            
                clf_dec=utilities.classification(X_train,y_train,X_test,c,"lr") 
                clf=clf_dec[0]      

                weightedGraphs=graph_utils.weightedGraphsFromFeatures(feat_docs_test,node_docs_test,clf,vectorizer)
                #remove connections using maximum spanning tree
                edmonds_test_graph=_graph=graph_utils.EdmondGraphs(weightedGraphs)
                #get predictions from edmond graph
                pred_labels=graph_utils.getPredictionsFromEdmond(feat_docs_test,node_docs_test,edmonds_test_graph,clf,vectorizer)      
                pred_labels=map(int, pred_labels)                
                
            if self.model=="transition":    
            
                clf=utilities.classify(X_train,y_train,c,"svm")          
                
                
                y_test=transition_utils.getLabelsFromGoldFile(node_docs_test,gold_docs_test)

                            
                pred_labels=transition_utils.getLabelsFromPredictions(node_docs_test,clf,vectorizer)

            # write the results to file
            utilities.printResultsToFile(y_test,pred_labels,self.results_dir+"/results.txt",c,self.model,randomSeed)
                
            
            
            

            