import findspark
findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

try:
    # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
    sc = ps.SparkContext('local[4]')
    print("Just created a SparkContext")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")
# Load and parse the data
data = sc.textFile("spark_sample_data/sample_lda_data.txt")

parsedData = data.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))

# Index documents with unique IDs
corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
print(corpus.take(10))

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=3, optimizer='online')

# Output topics. Each is a distribution over words (matching word count vectors)
print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize())
      + " words):")
topics = ldaModel.topicsMatrix()
for topic in range(3):
    print("Topic " + str(topic) + ":")
    for word in range(0, ldaModel.vocabSize()):
        print(" " + str(topics[word][topic]))
