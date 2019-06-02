import numpy

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating

if __name__ == "__main__":
	sc = SparkContext(appName="MovieLens-recommender")
	
	data = sc.textFile("/data/ml-latest-small/ratings.csv")
	
