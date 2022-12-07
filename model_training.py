import sys
from pyspark import SparkConf, SparkContext, SQLContext
from pyexpat import model
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.classification import DecisionTreeClassifier


if(len(sys.argv)==1):
        raise Exception("Execute script by providing proper parameters => model_train.py <dataset location> <output path>")

print("Starting Spark Connection")
conf = (SparkConf().setAppName("WineQuality-Training"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

dt_trained_model_result = sys.argv[2]+"dt-trained.model"  
rf_trained_model_result = sys.argv[2]+"rf-trained.model"
output_file = sys.argv[2]+"fitness_file.txt"
train_dataset = sys.argv[1]

print(f"Loading Training data from {train_dataset} ..")
df_training = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(train_dataset)

features = df_training.columns

va = VectorAssembler(inputCols=features, outputCol="features")
df_training = va.transform(df_training)

def decision_tree_training(file):
    print("===================Decision Tree Classifier model===================")
    file.write("===================Decision Tree Classifier model===================\n")
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = features[-1], maxDepth =2)
    dt_Model = dt.fit(df_training)
    dt_predictions = dt_Model.transform(df_training)
    print(f"Decision tree trained model Location {dt_trained_model_result} ..")
    dt_Model.write().overwrite().save(dt_trained_model_result)

    print("Evaluate the trained model...")
    file.write("Evaluate the trained model...\n")

    dt_evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="accuracy")
    dt_accuracy = dt_evaluator.evaluate(dt_predictions)
    print("Accuracy = %s" % (dt_accuracy))
    file.write("Accuracy = %s" % (dt_accuracy) + "\n")
    print("Test Error = %s" % (1.0 - dt_accuracy))
    file.write("Test Error = %s" % (1.0 - dt_accuracy) + "\n")

    dt_evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
    dt_f1score = dt_evaluator.evaluate(dt_predictions)
    print("Decision Tree F1-Score = %s" % (dt_f1score))
    file.write("Decision Tree F1-Score = %s" % (dt_f1score) + "\n")

def random_forest(file):
    print("===================Random Forest model===================")
    file.write("===================Random Forest model===================\n")
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = features[-1] , numTrees=60, maxBins=32, maxDepth=5, seed=42)
    rf_model = rf.fit(df_training)
    predictions = rf_model.transform(df_training)
    print(f"Random forest trained model Location : {rf_trained_model_result} ..")
    rf_model.write().overwrite().save(rf_trained_model_result)
    evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy = %s" % (accuracy))
    file.write("Accuracy = %s" % (accuracy) + "\n")
    print("Test Error = %s" % (1.0 - accuracy))
    file.write("Test Error = %s" % (1.0 - accuracy) + "\n")
    evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
    f1score = evaluator.evaluate(predictions)
    print("Random Forest F1-Score = %s" % (f1score))
    file.write("Random Forest F1-Score = %s" % (f1score) + "\n")


def main():
    with open(output_file, 'w') as file:
        decision_tree_training(file)
        random_forest(file)


if __name__ == "__main__":
    main()