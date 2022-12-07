import sys
from pyspark import SparkConf, SparkContext, SQLContext
from pyexpat import model
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.classification import DecisionTreeClassifier


if(len(sys.argv)==1):
    raise Exception("Execute script by providing proper parameters => model_predict.py <dataset location> <output path>")

conf = (SparkConf().setAppName("WineQuality-Prediction"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

dt_trained_model_result = sys.argv[2]+"dt-trained.model"  
rf_trained_model_result = sys.argv[2]+"rf-trained.model"
rf_predicted_model_result = sys.argv[2]+"rf-predicted.model"
dt_predicted_model_result = sys.argv[2]+"dt-predicted.model"
output_file = sys.argv[3]+"prediction_file.txt"
train_dataset = sys.argv[1] 

print(f"Loading data from {train_dataset}..")
df_validation = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(train_dataset)

features = df_validation.columns

features = [c for c in df_validation.columns if c != 'quality'] #Drop quality column


df_validation.select(features).describe().toPandas().transpose()


va = VectorAssembler(inputCols=features, outputCol="features")
df_validation = va.transform(df_validation)


def random_forest_prediction(file):
    print("===================Random Forest Prediction model===================")
    file.write("===================Random Forest Prediction model===================\n")
    rf_model = RandomForestClassificationModel.load(rf_trained_model_result)
    predictions = rf_model.transform(df_validation)
    print(f"Trained model Location: {rf_predicted_model_result} ..")
    rf_model.write().overwrite().save(rf_predicted_model_result)
    print("Evaluating the trained model...")
    evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy = %s" % (accuracy))
    print("Test Error = %s" % (1.0 - accuracy))
    file.write("Accuracy = %s" % (accuracy) + "\n")
    file.write("Test Error = %s" % (1.0 - accuracy) + "\n")

    evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
    f1score = evaluator.evaluate(predictions)
    print("F1-Score = %s" % (f1score))
    file.write("F1-Score = %s" % (f1score) + "\n")


def main():
    with open(output_file, 'w') as file:
        random_forest_prediction(file)


if __name__ == "__main__":
    main()
