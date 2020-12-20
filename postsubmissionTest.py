#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Author : Keval Kavle
#Date : 12-03-2020
#Subject : Cloud Computing CS643
#Assignment : Programming Assignment-2


# In[17]:


# Importing Libraries
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import ceil, col
import pyspark.sql.functions as func
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel
import sys


# In[3]:


#SparkConf().getAll()
##### Opening a Spark Session  #####
spark = SparkSession.builder.master("local").appName("WineQualityPrediction-CS643").getOrCreate()


# In[6]:


##### Test Dataset ####
testfile = 'TestDataset.csv'
valDf = spark.read.csv(testfile,header='true', inferSchema='true', sep=';') 


# In[12]:


#valDf.show(2)


# In[7]:


valDf = valDf.withColumnRenamed(valDf.columns[-1],'quality')


# In[8]:


#valDf.show(5)


# In[18]:


#train = trainTrans


# In[19]:


#trainTrans.show(2)


# In[20]:


#trainTrans.select("independentFeatures").show(2,truncate=False)


# In[23]:


#train.show(2,truncate=False)
#train.select("scaledFeatures").show(2,truncate=False)


# In[24]:


#train.show(2,truncate=False)


# In[10]:


valColumns = [c for c in valDf.columns  if "quality" not in c  ]
#valColumns


# In[11]:


valAssembler = VectorAssembler(inputCols=valColumns,outputCol="independentFeatures")


# In[12]:


valTrans = valAssembler.transform(valDf)


# In[13]:


#valTrans.select("independentFeatures").show(2,truncate=False)


# In[14]:


standardScaler = StandardScaler(inputCol="independentFeatures",outputCol="scaledFeatures")


# In[15]:


val = standardScaler.fit(valTrans).transform(valTrans)


# In[31]:


#val = valTrans


# In[23]:


model= RandomForestClassificationModel.load("rfModel")


# In[35]:


predictions = model.transform(val)
#predictions.show(2)


# In[36]:


predictions = predictions.withColumn("prediction", func.round("prediction"))


# In[37]:


##### Random Forest  Ends #####


# In[40]:



# In[50]:


#predResults.show()


# In[51]:


#val.show(2)


# In[52]:


#predictions.show(2)


# In[53]:


#roundoffPrediction =  predictions.select("prediction", func.round(col('prediction')))
#roundoffPrediction.show(2)


# In[54]:


#predictions = predictions.withColumn("predictionFinal", roundoffPrediction[1])
#predictions = predictions.withColumn("prediction", func.round("prediction"))


# In[55]:


#predictions.show()


# In[56]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[38]:


####### Random Forest Accuarcy - not using this as simple linear regression is giving better results ############
evalAcc = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evalAcc.evaluate(predictions)

print("accuracy Test Error = %g" % (1.0 - accuracy))


transformed_data = model.transform(val)
transformed_data = transformed_data.withColumn("prediction", func.round("prediction"))
print(evalAcc.getMetricName(), 'accuracy:', evalAcc.evaluate(transformed_data))


# In[39]:


####### Random Forest f1 - not using this as simple linear regression is giving better results ############
evalVal = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="f1")
accuracy = evalVal.evaluate(predictions)
print("f1 Test Error = %g" % (1.0 - accuracy))
transformed_data = model.transform(val)
transformed_data = transformed_data.withColumn("prediction", func.round("prediction"))
print(evalVal.getMetricName(), 'accuracy :', evalVal.evaluate(transformed_data))


# In[45]:


####### Linear Regression Accuarcy and f1 ############


# Create evaluator
#evaluatorMulti = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")

# Make predicitons
#predictionAndTarget = regressor.transform(val).select("quality", "prediction")
#predictionAndTarget = predictionAndTarget.withColumn("prediction", func.round("prediction"))
# Get metrics
#acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
#f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})


# In[46]:


##print("Accuracy :" , acc * 100 , "%")
##print("f1 Score :" , f1)


# In[ ]:




