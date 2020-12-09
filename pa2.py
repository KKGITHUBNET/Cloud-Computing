#!/usr/bin/env python
# coding: utf-8

# In[153]:


#Author : Keval Kavle
#Date : 12-03-2020
#Subject : Cloud Computing CS643
#Assignment : Programming Assignment-2


# In[131]:


# Importing Libraries
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import ceil, col
import pyspark.sql.functions as func
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys


# In[155]:


#SparkConf().getAll()
##### Opening a Spark Session  #####
spark = SparkSession.builder.master("local").appName("WineQualityPrediction-CS643").getOrCreate()


# In[156]:


#### Training Dataset
trainDf = spark.read.csv('TrainingDataset.csv',header='true', inferSchema='true', sep=';')


# In[210]:


from pyspark.sql.functions import isnan, when, count, col

#trainDf.select([count(when(isnan(c), c)).alias(c) for c in trainDf.columns]).show()


# In[157]:


#trainDf


# In[159]:


#trainDf.printSchema()


# In[160]:


#trainDf.show(2)


# In[161]:


from pyspark.sql.types import *

schemaTrain = trainDf.schema


# In[162]:


##### Test Dataset ####
if (sys.argv[1] == '' or sys.argv[1] == "-f"):
    testpath = "./"
else :
    testpath = sys.argv[1]

print(testpath)
#testpath = "C:/Users/DELL/Python_Practice/CS643-ProgrammingAssignment2/"
testfile = testpath + 'TestDataset.csv'
valDf = spark.read.csv(testfile,header='true', schema=schemaTrain, sep=';') 


# In[206]:


#valDf.show(2)


# In[208]:


#valDf.show(5)


# In[166]:


trainColumns = [c for c in trainDf.columns  if "quality" not in c  ]
#trainColumns


# In[167]:


trainAssembler = VectorAssembler(inputCols=trainColumns,outputCol="independentFeatures")


# In[168]:


trainTrans = trainAssembler.transform(trainDf)


# In[169]:


train = trainTrans


# In[170]:


#trainTrans.show(2)


# In[171]:


#trainTrans.select("independentFeatures").show(2,truncate=False)


# In[172]:


#standardScaler = StandardScaler(inputCol="independentFeatures",outputCol="scaledFeatures")


# In[173]:


#train = standardScaler.fit(trainTrans).transform(trainTrans)


# In[174]:


#train.show(2,truncate=False)
#train.select("scaledFeatures").show(2,truncate=False)


# In[175]:


#train.show(2,truncate=False)


# In[176]:


valColumns = [c for c in valDf.columns  if "quality" not in c  ]
#valColumns


# In[177]:


valAssembler = VectorAssembler(inputCols=valColumns,outputCol="independentFeatures")


# In[178]:


valTrans = valAssembler.transform(valDf)


# In[179]:


#valTrans.select("independentFeatures").show(2,truncate=False)


# In[180]:


#standardScaler = StandardScaler(inputCol="independentFeatures",outputCol="scaledFeatures")


# In[181]:


#val = standardScaler.fit(valTrans).transform(valTrans)


# In[182]:


val = valTrans


# In[132]:


##### Random Forest  #####
rf = RandomForestRegressor(labelCol="quality", maxDepth=3,
    maxBins=10,featuresCol="independentFeatures",numTrees=5)


# In[184]:


model = rf.fit(train)


# In[185]:


model.write().overwrite().save("rfModel")


# In[186]:


predictions = model.transform(val)
#predictions.show(2)


# In[134]:


predictions = predictions.withColumn("prediction", func.round("prediction"))
predictions.show(2)


# In[188]:


##### Random Forest  Ends #####


# In[195]:


#predResults.show()


# In[196]:


#val.show(2)


# In[197]:


#predictions.show(2)


# In[198]:


#roundoffPrediction =  predictions.select("prediction", func.round(col('prediction')))
#roundoffPrediction.show(2)


# In[199]:


#predictions = predictions.withColumn("predictionFinal", roundoffPrediction[1])
#predictions = predictions.withColumn("prediction", func.round("prediction"))


# In[200]:


#predictions.show()


# In[201]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[135]:



# Create evaluator
evaluatorMulti = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction")

# Make predicitons
predictionAndTarget = model.transform(val).select('""""quality"""""', "prediction")
predictionAndTarget = predictionAndTarget.withColumn("prediction", func.round("prediction"))
# Get metrics
acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})


# In[138]:


print(acc)
print(f1)


# In[ ]:




