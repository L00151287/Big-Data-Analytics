# Databricks notebook source
# MAGIC %md 
# MAGIC ## Big Data Analytics Technical Project

# COMMAND ----------

# MAGIC %md 
# MAGIC ####Preprocessing

# COMMAND ----------

#### Importing needful libraries

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer


# COMMAND ----------

###The entry point into all functionality in Spark is the SparkSession class.To create a basic SparkSession, just use SparkSession.builder

spark = SparkSession \
    .builder \
    .appName("Spark ML on titanic data ") \
    .getOrCreate()

# COMMAND ----------

##Loading CSV data
dataset = "/FileStore/tables/train.csv"
titanic_df = spark.read.csv(dataset,header = 'True',inferSchema='True')
display(titanic_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Analyzing the dataset

# COMMAND ----------

###The below is the schema of data
titanic_df.printSchema()

# COMMAND ----------

passengers_count = titanic_df.count()
print(passengers_count)

# COMMAND ----------

###Viewing few rows
titanic_df.show(5)

# COMMAND ----------

# MAGIC %md Summary of data

# COMMAND ----------

titanic_df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC  Checking Schema of the dataset

# COMMAND ----------

titanic_df.printSchema()

# COMMAND ----------

# MAGIC %md ###   Exploratory data analysis (EDA)

# COMMAND ----------

# MAGIC %md 
# MAGIC  Selecting few features

# COMMAND ----------

###Checking survival rate using feature class.

titanic_df.select("Survived","Pclass","Embarked").show()

# COMMAND ----------

groupBy_output = titanic_df.groupBy("Survived", "Pclass").count()

# COMMAND ----------

display(groupBy_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Here it can be seen that the Pclass1 people were given priority to pclass3 people, even though
# MAGIC We can clearly see that Passenegers Of Pclass 1 were given a very high priority while rescue. Even though the the number of Passengers in Pclass 3 were a lot higher, still the number of survival from them is very low.

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Knowing the number of Passengers Survived ?

# COMMAND ----------

titanic_df.groupBy("Survived").count().show()

# COMMAND ----------

# MAGIC %md Out of 891 passengers in dataset, only about 342 survived.

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### To know the particulars about survivors we have to explore more of the data.
# MAGIC ###### The survival rate can be determined by different features of the dataset such as Sex, Port of Embarcation, Age; few to be mentioned.

# COMMAND ----------

###Checking survival rate using feature Sex.

titanic_df.groupBy("Sex","Survived").count().show()

# COMMAND ----------

grp_output = titanic_df.groupBy( "Sex", "Survived").count()

# COMMAND ----------

display(grp_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Although the number of males are more than females on ship, the female survivors are twice the number of males saved.

# COMMAND ----------

### Checking total number of passengers in each Pclass survived.

a = titanic_df.groupBy("Pclass").count()

# COMMAND ----------

display(a)

# COMMAND ----------

### checking Age with feature pclass.
b = titanic_df.groupBy("Age", "Survived").count()

# COMMAND ----------

display(b)

# COMMAND ----------

# MAGIC %md #### Handling Null values

# COMMAND ----------

# This function use to print feature with null values and null count 
def null_value_count(df):
  null_columns_counts = []
  numRows = df.count()
  for k in df.columns:
    nullRows = df.where(col(k).isNull()).count()
    if(nullRows > 0):
      temp = k,nullRows
      null_columns_counts.append(temp)
  return(null_columns_counts)

# Calling function
null_columns_count_list = null_value_count(titanic_df)
spark.createDataFrame(null_columns_count_list, ['Column_With_Null_Value', 'Null_Values_Count']).show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md Age feature has 177 null values.

# COMMAND ----------

mean_age = titanic_df.select(mean('Age')).collect()[0][0]
print(mean_age)

# COMMAND ----------

titanic_df.select("Name").show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ######  To replace these NaN values, we can assign them the mean age of the dataset.But the problem is, there were many people with many different ages. We just cant assign a 4 year kid with the mean age that is 29 years. 

# COMMAND ----------

# MAGIC %md ###### we can check the Name feature. Looking upon the feature, we can see that the names have a salutation like Mr or Mrs. Thus we can assign the mean values of Mr and Mrs to the respective groups

# COMMAND ----------

###Using the Regex ""[A-Za-z]+)." we extract the initials from the Name. It looks for strings which lie between A-Z or a-z and followed by a .(dot).

titanic_df = titanic_df.withColumn("Initial",regexp_extract(col("Name"),"([A-Za-z]+)\.",1))


# COMMAND ----------


titanic_df.show()

# COMMAND ----------

titanic_df.select("Initial").distinct().show()


# COMMAND ----------

### There are some misspelled Initials like Mlle or Mme that stand for Miss. I will replace them with Miss and same thing for other values.

titanic_df = titanic_df.replace(['Mlle','Mme', 'Ms', 'Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
               ['Miss','Miss','Miss','Mr','Mr',  'Mrs',  'Mrs',  'Other',  'Other','Other','Mr','Mr','Mr'])


# COMMAND ----------

titanic_df.select("Initial").distinct().show()


# COMMAND ----------

###lets check the average age by Initials
titanic_df.groupby('Initial').avg('Age').collect()

# COMMAND ----------

###Let's impute missing values in age feature based on average age of Initials

titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Miss") & (titanic_df["Age"].isNull()), 22).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Other") & (titanic_df["Age"].isNull()), 46).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Master") & (titanic_df["Age"].isNull()), 5).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Mr") & (titanic_df["Age"].isNull()), 33).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Mrs") & (titanic_df["Age"].isNull()), 36).otherwise(titanic_df["Age"]))


# COMMAND ----------

# MAGIC %md
# MAGIC Check the imputation 

# COMMAND ----------

###Check the imputation

titanic_df.filter(titanic_df.Age==46).select("Initial").show()


# COMMAND ----------

titanic_df.select("Age").show()

# COMMAND ----------

###Embarked feature has only two missining values. Let's check values within Embarked
titanic_df.groupBy("Embarked").count().show()

# COMMAND ----------

###Majority Passengers boarded from "S". We can impute with "S"
titanic_df = titanic_df.na.fill({"Embarked" : 'S'})


# COMMAND ----------

###We can drop Cabin features as it has lots of null values
titanic_df = titanic_df.drop("Cabin")

# COMMAND ----------

titanic_df.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC We can create a new feature called "Family_size" and "Alone" and analyse it. This feature is the summation of Parch(parents/children) and SibSp(siblings/spouses). It gives us a combined data so that we can check if survival rate have anything to do with family size of the passengers

# COMMAND ----------

titanic_df = titanic_df.withColumn("Family_Size",col('SibSp')+col('Parch'))

# COMMAND ----------

titanic_df.groupBy("Family_Size").count().show()

# COMMAND ----------

ab = titanic_df.groupBy("Family_Size").count()
display(ab)

# COMMAND ----------

titanic_df = titanic_df.withColumn('Alone',lit(0))
titanic_df = titanic_df.withColumn("Alone",when(titanic_df["Family_Size"] == 0, 1).otherwise(titanic_df["Alone"]))


# COMMAND ----------

titanic_df.columns

# COMMAND ----------


#convert Sex, Embarked & Initial columns from string to number using StringIndexer.

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(titanic_df) for column in ["Sex","Embarked","Initial"]]
pipeline = Pipeline(stages=indexers)
titanic_df = pipeline.fit(titanic_df).transform(titanic_df)

# COMMAND ----------

titanic_df.show()

# COMMAND ----------

titanic_df.printSchema()

# COMMAND ----------

###Drop columns which are not required

titanic_df = titanic_df.drop("PassengerId","Name","Ticket","Cabin","Embarked","Sex","Initial")

# COMMAND ----------

titanic_df.show()

# COMMAND ----------

# MAGIC %md ###### Let's put all features into vector

# COMMAND ----------

feature = VectorAssembler(inputCols=titanic_df.columns[1:],outputCol="features")
feature_vector= feature.transform(titanic_df)

# COMMAND ----------

feature_vector.show()

# COMMAND ----------

###Now that the data is all set, let's split it into training and test. I'll be using 80% of it.

(trainingData, testData) = feature_vector.randomSplit([0.8, 0.2],seed = 11)

# COMMAND ----------

# MAGIC %md ### Modelling 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ######  Classification Algorithms used to model the dataset are shown below
# MAGIC 
# MAGIC LogisticRegression
# MAGIC 
# MAGIC DecisionTreeClassifier
# MAGIC 
# MAGIC RandomForestClassifier
# MAGIC 
# MAGIC Gradient-boosted tree classifier
# MAGIC 
# MAGIC NaiveBayes
# MAGIC 
# MAGIC Support Vector Machine

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### LogisticRegression

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="Survived", featuresCol="features")
#Training algo
lrModel = lr.fit(trainingData)
lr_prediction = lrModel.transform(testData)
lr_prediction.select("prediction", "Survived", "features").show()
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Evaluating accuracy of LogisticRegression.

# COMMAND ----------

lr_accuracy = evaluator.evaluate(lr_prediction)
print("Accuracy of LogisticRegression is = %g"% (lr_accuracy))
print("Test Error of LogisticRegression = %g " % (1.0 - lr_accuracy))

# COMMAND ----------

display(lr_prediction)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### DecisionTreeClassifier

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
dt_model = dt.fit(trainingData)
dt_prediction = dt_model.transform(testData)
dt_prediction.select("prediction", "Survived", "features").show()


# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Evaluating accuracy of DecisionTreeClassifier.

# COMMAND ----------

dt_accuracy = evaluator.evaluate(dt_prediction)
print("Accuracy of DecisionTreeClassifier is = %g"% (dt_accuracy))
print("Test Error of DecisionTreeClassifier = %g " % (1.0 - dt_accuracy))


# COMMAND ----------

display(dt_prediction)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### RandomForestClassifier

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
rf_model = rf.fit(trainingData)
rf_prediction = rf_model.transform(testData)
rf_prediction.select("prediction", "Survived", "features").show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Evaluating accuracy of RandomForestClassifier.

# COMMAND ----------

rf_accuracy = evaluator.evaluate(rf_prediction)
print("Accuracy of RandomForestClassifier is = %g"% (rf_accuracy))
print("Test Error of RandomForestClassifier  = %g " % (1.0 - rf_accuracy))

# COMMAND ----------

display(rf_prediction)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Gradient-boosted tree classifier

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="Survived", featuresCol="features",maxIter=10)
gbt_model = gbt.fit(trainingData)
gbt_prediction = gbt_model.transform(testData)
gbt_prediction.select("prediction", "Survived", "features").show()


# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Evaluate accuracy of Gradient-boosted.

# COMMAND ----------

gbt_accuracy = evaluator.evaluate(gbt_prediction)
print("Accuracy of Gradient-boosted tree classifie is = %g"% (gbt_accuracy))
print("Test Error of Gradient-boosted tree classifie %g"% (1.0 - gbt_accuracy))


# COMMAND ----------

display(gbt_prediction)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### NaiveBayes

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(labelCol="Survived", featuresCol="features")
nb_model = nb.fit(trainingData)
nb_prediction = nb_model.transform(testData)
nb_prediction.select("prediction", "Survived", "features").show()


# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Evaluating accuracy of NaiveBayes.

# COMMAND ----------

nb_accuracy = evaluator.evaluate(nb_prediction)
print("Accuracy of NaiveBayes is  = %g"% (nb_accuracy))
print("Test Error of NaiveBayes  = %g " % (1.0 - nb_accuracy))

# COMMAND ----------

display(nb_prediction)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Support Vector Machine

# COMMAND ----------

from pyspark.ml.classification import LinearSVC
svm = LinearSVC(labelCol="Survived", featuresCol="features")
svm_model = svm.fit(trainingData)
svm_prediction = svm_model.transform(testData)
svm_prediction.select("prediction", "Survived", "features").show()


# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Evaluating the accuracy of Support Vector Machine.

# COMMAND ----------

svm_accuracy = evaluator.evaluate(svm_prediction)
print("Accuracy of Support Vector Machine is = %g"% (svm_accuracy))
print("Test Error of Support Vector Machine = %g " % (1.0 - svm_accuracy))

# COMMAND ----------

display(svm_prediction)
