import time
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics

def parseInput(line):
	fields = line.value.split(',')
	return (int(fields[0]), int(fields[1]), float(fields[2]))

if __name__ == "__main__":
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
	print("Data loaded!")
	
	# Build the recommendation model using Alternating Least Squares
	print("Training the model...")
	start_train = time.time()
	rank = 10
	iterations = 10
	model = ALS.train(ratingsRDD, rank, iterations)
	end_train = time.time()
	
	print("Model trained in " + str(end_train - start_train) +"s!")
	
	print("Building predictions...")
	# Create tuple (userId, movieId)
	testdata = ratingsRDD.map(lambda p: (p[0], p[1]))
	# Predict and output tuple ((userId, movieId), prediction)
	start_pred = time.time()
	predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
	end_pred = time.time()
	# Join prediction to ratings: output tuple ((userId, movieId), (rate, prediction))
	ratesAndPreds = ratingsRDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	print("Predictions builded in " + str(end_pred - start_pred) + "s!")
	
	# Recommend 3 movies for userId=1
	user1_recomm = model.recommendProducts(1, 3)
	print(user1_recomm)
	
	# Evaluate using only tuple (rate, prediction)
	print("Evaluating the model...")
	metrics = RegressionMetrics(ratesAndPreds.map(lambda t: t[1]))
	rmse = metrics.rootMeanSquaredError
	print("Root Mean Squared Error = " + str(rmse))
	
	spark.stop()
