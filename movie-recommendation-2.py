import numpy
import time
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def parseInput(line):
	fields = line.value.split(',')
	return Row(userId=int(fields[0]), movieId=int(fields[1]), rating=float(fields[2]), timestamp=long(fields[3]))


if __name__ == "__main__":
	# Build new session or retrieve existing one
	print("Creating session...")
	spark = SparkSession.builder\
		.master("local[*]")\
		.appName("movielens-recommender-2")\
		.getOrCreate()
	
	ml_small_path = "data/ml-latest-small/ratings.csv"
	
	print("Loading data from " + ml_small_path + "...")
	dataRDD = spark.read.text(ml_small_path).rdd
	dataRDD_header = dataRDD.take(1)[0]
	dataRDD_filtered = dataRDD.filter(lambda l: l!=dataRDD_header)
	ratingsRDD = dataRDD_filtered.map(parseInput)
	ratings = spark.createDataFrame(ratingsRDD).cache()
	print("Data loaded!")
	
	print("Training the model...")
	start_train = time.time()
	# Build the recommendation model using Alternating Least Squares
	rank = 10
	iterations = 10
	als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=50, regParam=0.01, nonnegative=False)
	model = als.fit(ratings)
	end_train = time.time()
	print("Model trained in " + str(end_train - start_train) + "s!")
	
	# Build predictions on 'test' subset of 'ratings'
	print("Building predictions...")
	start_pred = time.time()
	predictions = model.transform(ratings)
	end_pred = time.time()
	print("Predictions builded in " + str(end_pred - start_pred) + "s!")
	
	# Recommend 3 movies for userId=1
	user1_subset = ratings.where(ratings.userId==1)
	user1_recomm = model.recommendForUserSubset(user1_subset, 3)
	print(user1_recomm.take(3))
	
	# Evaluate the model on training data
	print("Evaluating the model...")
	evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
		predictionCol="prediction")
	rmse = evaluator.evaluate(predictions)
	print("Root Mean Squared Error = " + str(rmse))
	
	spark.stop()
