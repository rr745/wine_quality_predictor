from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .getOrCreate()

# File paths
training_data_path = "/home/ec2-user/wine_quality_predictor/TrainingDataset.csv"
validation_data_path = "/home/ec2-user/wine_quality_predictor/ValidationDataset.csv"
model_save_path = "/home/ec2-user/wine_quality_model"

# Read datasets
train_df = spark.read.csv(training_data_path, header=True, inferSchema=True)
valid_df = spark.read.csv(validation_data_path, header=True, inferSchema=True)

# Prepare data for training
feature_columns = train_df.columns[:-1]  # Exclude the label column
label_column = train_df.columns[-1]  # Assume the last column is the label

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_data = assembler.transform(train_df).select("features", label_column)
valid_data = assembler.transform(valid_df).select("features", label_column)

# Train a Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol=label_column, maxIter=10)

# Hyperparameter tuning using CrossValidator
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = MulticlassClassificationEvaluator(
    labelCol=label_column,
    predictionCol="prediction",
    metricName="f1"
)

cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
cv_model = cv.fit(train_data)

# Validate the model
predictions = cv_model.transform(valid_data)
f1_score = evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score}")

# Save the model
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

cv_model.bestModel.write().overwrite().save(model_save_path)
print(f"Model saved to {model_save_path}")

# Stop the Spark session
spark.stop()
