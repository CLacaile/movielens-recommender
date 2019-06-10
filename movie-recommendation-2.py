import time
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Correlation

def parseInput(line):
	fields = line.value.split(',')
	return Row(userId=int(fields[0]), movieId=int(fields[1]), rating=float(fields[2]), timestamp=long(fields[3]))


if __name__ == "__main__":
	# Build new session or retrieve existing one
	print("-> Creating session...")
	spark = SparkSession.builder\
		.master("local[*]")\
		.appName("movielens-recommender-2")\
		.getOrCreate()
	
	ml_small_path = "data/ml-latest-small/ratings.csv"
	
	print("-> Loading data from " + ml_small_path + "...")
	dataRDD = spark.read.text(ml_small_path).rdd
	dataRDD_header = dataRDD.take(1)[0]
	dataRDD_filtered = dataRDD.filter(lambda l: l!=dataRDD_header)
	ratingsRDD = dataRDD_filtered.map(parseInput)
	ratings = spark.createDataFrame(ratingsRDD).cache()
	print("Data loaded!")
	
	print("-> Splitting into a training set and a testing set...")
	(training, testing) = ratings.randomSplit([0.8, 0.2], seed=0L)
	
	print("-> Training the model...")
	start_train = time.time()
	# Build the recommendation model using Alternating Least Squares
	rank = 10
	iterations = 10
	als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=20, regParam=0.2, nonnegative=True, coldStartStrategy="drop")
	model = als.fit(training)
	end_train = time.time()
	print("Model trained in " + str(end_train - start_train) + "s!")
	
	# Build predictions on 'test' subset of 'ratings'
	print("-> Building predictions on the testset...")
	start_pred = time.time()
	predictions_test = model.transform(testing)
	end_pred = time.time()
	print("Predictions built in " + str(end_pred - start_pred) + "s!")
	
	# Build predictions on 'train' subset of 'ratings'
	print("-> Building predictions on the trainset...")
	start_pred = time.time()
	predictions_train = model.transform(training)
	end_pred = time.time()
	print("Predictions built in " + str(end_pred - start_pred) + "s!")
	
	# Recommend 3 movies for userId=1
	#user1_subset = ratings.where(ratings.userId==1)
	#user1_recomm = model.recommendForUserSubset(user1_subset, 3)
	
	# Evaluate the model
	print("-> Evaluating the model...")
	evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
		predictionCol="prediction")
	rmse_test = evaluator.evaluate(predictions_test)
	print("RMSE testset = " + str(rmse_test))
	rmse_train = evaluator.evaluate(predictions_train)
	print("RMSE trainset = " + str(rmse_train))
	print("Ratio RMSE_test/RMSE_train = " + str(rmse_test/rmse_train))
	
	#spearman = Correlation.corr(predictions, "prediction", "spearman").head()
	#print("Spearman correlation matrix:\n", str(spearman[0]))
	
	spark.stop()
