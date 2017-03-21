import pipeline

 
 
#script to execute the pipeline of the Malibu project
if __name__ == "__main__":
       
       configFile="config/config.properties"   
       graphFeaturesFile="config/graph_features.properties"   
      
       c_array=[1e0,1e1,1e2,1e3,1e4]
       randomSeed=42
       models=["transition","mtt","threshold","edmond"]
       for model in models:
           
           pip=pipeline.pipeline(configFile,graphFeaturesFile,model)
             
           for c in c_array:
                              
                    pip.computeScores(c,randomSeed)     
                    
                  
       