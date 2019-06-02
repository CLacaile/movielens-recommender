import numpy

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating

if __name__ == "__main__":
	sc = SparkContext(appName="MovieLens-recommender")
	
	ml_small_path = "data/ml-latest-small/ratings.csv"
	ml_path = "data/ml-latest/ratings.csv"
	
	print("Loading data from " + ml_small_path)
	data = sc.textFile(ml_small_path)
	data_header = data.take(1)[0]
	data_filtered = data.filter(lambda l: l!=data_header)
	ratings = data_filtered.map(lambda l: l.split(','))\
		.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
		
	# Build the recommendation model using Alternating Least Squares
	model = ALS.train(ratings, 10, 10)
	
	# Evaluate the model on training data
	testdata = ratings.map(lambda p: (p[0], p[1]))
	predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
	ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
	print("Mean Squared Error = " + str(MSE))
