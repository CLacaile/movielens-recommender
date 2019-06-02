import numpy

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def parseInput(line):
	fields = line.value.split(',')
	return Row(userId=int(fields[0]), movieId=int(fields[1]), rating=float(fields[2]), timestamp=long(fields[3]))


if __name__ == "__main__":
	spark = SparkSession.builder.appName("movielens-recommender-2").getOrCreate()
	
	
	ml_small_path = "data/ml-latest-small/ratings.csv"
	ml_path = "data/ml-latest/ratings.csv"
	
	print("Loading data from " + ml_small_path)
	data = spark.read.text(ml_small_path).rdd
	data_header = data.take(1)[0]
	data_filtered = data.filter(lambda l: l!=data_header)
	ratingsRDD = data_filtered.map(parseInput)
	ratings = spark.createDataFrame(ratingsRDD).cache()
	
	(training, test) = ratings.randomSplit([0.8, 0.2])
		
	# Build the recommendation model using Alternating Least Squares
	als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
		coldStartStrategy="drop")
	model = als.fit(ratings)
	
	# Evaluate the model on training data
	predictions = model.transform(test)
	evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
		predictionCol="prediction")
	rmse = evaluator.evaluate(predictions)
	print("Mean Squared Error = " + str(rmse*rmse))
