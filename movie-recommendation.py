import time
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RegressionMetrics

if __name__ == "__main__":
	print("Creating context...")
	sc = SparkContext(appName="MovieLens-recommender")
	
	ml_small_path = "data/ml-latest-small/ratings.csv"
	ml_path = "data/ml-latest/ratings.csv"
	
	print("Loading data from " + ml_small_path + "...")
	data = sc.textFile(ml_small_path)
	data_header = data.take(1)[0]
	data_filtered = data.filter(lambda l: l!=data_header)
	ratings = data_filtered.map(lambda l: l.split(','))\
		.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
	print("Data loaded!")
	
	# Build the recommendation model using Alternating Least Squares
	print("Training the model...")
	start_train = time.time()
	rank = 10
	iterations = 10
	model = ALS.train(ratings, rank, iterations)
	end_train = time.time()
	
	print("Model trained in " + str(end_train - start_train) +"s!")
	
	print("Building predictions...")
	# Create tuple (userId, movieId)
	testdata = ratings.map(lambda p: (p[0], p[1]))
	# Predict and output tuple ((userId, movieId), prediction)
	start_pred = time.time()
	predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
	end_pred = time.time()
	# Join prediction to ratings: output tuple ((userId, movieId), (rate, prediction))
	ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	print("Predictions builded in " + str(end_pred - start_pred) + "s!")
	
	# Evaluate using only tuple (rate, prediction)
	print("Evaluating the model...")
	metrics = RegressionMetrics(ratesAndPreds.map(lambda t: t[1]))
	rmse = metrics.rootMeanSquaredError
	print("Root Mean Squared Error = " + str(rmse))
