#!/usr/bin/env python
# coding: utf-8

# In[74]:


#Author : Keval Kavle
#Date : 12-03-2020
#Subject : Cloud Computing CS643
#Assignment : Programming Assignment-2


# In[75]:


# Importing Libraries
#import findspark
#findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import ceil, col
import pyspark.sql.functions as func
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

# In[76]:


#SparkConf().getAll()
##### Opening a Spark Session  #####
spark = SparkSession.builder.master("local").appName("WineQualityPrediction-CS643").getOrCreate()


# In[77]:


#### Training Dataset
trainDf = spark.read.csv('TrainingDataset.csv',header='true', inferSchema='true', sep=';')

from pyspark.sql.functions import isnan, when, count, col

#trainDf.select([count(when(isnan(c), c)).alias(c) for c in trainDf.columns]).show()
# In[78]:


#trainDf


# In[79]:


#trainDf.show(5,truncate=False)
trainDf = trainDf.withColumnRenamed(trainDf.columns[-1],'quality')


# In[80]:


#trainDf.printSchema()


# In[81]:


#trainDf.show(2)


# In[82]:

# 

##### Test Dataset #####
from pyspark.sql.types import *

schemaTrain = trainDf.schema

if (sys.argv[1] == 1):
    testpath = ""
else :
    testpath = sys.argv[1]


testfile = testpath + "TestDataset.csv"
valDf = spark.read.csv(testfile,header='true',schema=schemaTrain , sep=';') 
#valDf.show(2)
# In[83]:


valDf = valDf.withColumnRenamed(valDf.columns[-1],'quality')


# In[84]:


#valDf.show(5)


# In[85]:


trainColumns = [c for c in trainDf.columns  if "quality" not in c  ]
#trainColumns


# In[86]:


trainAssembler = VectorAssembler(inputCols=trainColumns,outputCol="independentFeatures")


# In[87]:


trainTrans = trainAssembler.transform(trainDf)


# In[88]:


#train = trainTrans


# In[89]:


#trainTrans.show(2)


# In[90]:


#trainTrans.select("independentFeatures").show(2,truncate=False)


# In[91]:


standardScaler = StandardScaler(inputCol="independentFeatures",outputCol="scaledFeatures")


# In[92]:


train = standardScaler.fit(trainTrans).transform(trainTrans)


# In[93]:


#train.show(2,truncate=False)
#train.select("scaledFeatures").show(2,truncate=False)


# In[94]:


#train.show(2,truncate=False)


# In[95]:


valColumns = [c for c in valDf.columns  if "quality" not in c  ]
#valColumns


# In[96]:


valAssembler = VectorAssembler(inputCols=valColumns,outputCol="independentFeatures")


# In[97]:


valTrans = valAssembler.transform(valDf)


# In[98]:


#valTrans.select("independentFeatures").show(2,truncate=False)


# In[99]:


standardScaler = StandardScaler(inputCol="independentFeatures",outputCol="scaledFeatures")


# In[100]:

#valTrans.show(2)
val = standardScaler.fit(valTrans).transform(valTrans)


# In[101]:


#val = valTrans


# In[102]:


##### Random Forest  #####
rf = RandomForestClassifier(labelCol="quality", featuresCol="independentFeatures",numTrees=10)


# In[103]:


model = rf.fit(train)


# In[104]:


model.write().overwrite().save("rfModel")


# In[105]:


predictions = model.transform(val)
#predictions.show(2)


# In[106]:


predictions = predictions.withColumn("prediction", func.round("prediction"))


# In[107]:


##### Random Forest  Ends #####


# In[108]:


#### Linear Regression  #####
regressor = LinearRegression(featuresCol="independentFeatures",labelCol="quality")
regressor=regressor.fit(train)


# In[109]:


predResults = regressor.evaluate(val)


# In[110]:


predResults = predResults.predictions


# In[111]:


regressor.write().overwrite().save("lrModel")


# In[112]:


predResults = predResults.withColumn("prediction", func.round("prediction"))
#predResults.show(2)


# In[113]:


##### Linear Regression Ends ######


# In[114]:


#predResults.show()


# In[115]:


#val.show(2)


# In[116]:


#predictions.show(2)


# In[117]:


#roundoffPrediction =  predictions.select("prediction", func.round(col('prediction')))
#roundoffPrediction.show(2)


# In[118]:


#predictions = predictions.withColumn("predictionFinal", roundoffPrediction[1])
#predictions = predictions.withColumn("prediction", func.round("prediction"))


# In[119]:


#predictions.show()


# In[120]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[121]:


####### Random Forest Accuarcy - not using this as simple linear regression is giving better results ############
evalAcc = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evalAcc.evaluate(predictions)

##print("accuracy Test Error = %g" % (1.0 - accuracy))


transformed_data = model.transform(val)
transformed_data = transformed_data.withColumn("prediction", func.round("prediction"))
##print(evalAcc.getMetricName(), 'accuracy:', evalAcc.evaluate(transformed_data))


# In[122]:


####### Random Forest f1 - not using this as simple linear regression is giving better results ############
evalVal = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="f1")
accuracy = evalVal.evaluate(predictions)
##print("f1 Test Error = %g" % (1.0 - accuracy))
transformed_data = model.transform(val)
transformed_data = transformed_data.withColumn("prediction", func.round("prediction"))
##print(evalVal.getMetricName(), 'accuracy :', evalVal.evaluate(transformed_data))


# In[123]:


####### Linear Regression Accuarcy and f1 ############


# Create evaluator
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")

# Make predicitons
predictionAndTarget = regressor.transform(val).select("quality", "prediction")
predictionAndTarget = predictionAndTarget.withColumn("prediction", func.round("prediction"))
# Get metrics
acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})


# In[124]:

print("##### Testdataset Accuracy  #####")
print("Accuracy :" , acc * 100 , "%")
print("f1 Score :" , f1)

