import findspark
findspark.init()
import pyspark as ps
from pyspark.sql import SQLContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.sql import functions as F
import warnings

try:
    # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
    sc = ps.SparkContext('local[4]')
    print("Just created a SparkContext")
    sqlContext = SQLContext(sc)
except ValueError:
    warnings.warn("SparkContext already exists in this scope")
    sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').options(header='false').load('sentiment140dataset/training.1600000.processed.noemoticon.csv')
print(type(df))
df.show(5)

columns_to_drop = ['_c0', '_c3']
df = df.drop(*columns_to_drop)
df = df.withColumnRenamed('_c1', 'tweet_id').withColumnRenamed('_c2', 'date').withColumnRenamed('_c4', 'username').withColumnRenamed('_c5', 'tweet')
print(df.show(5))

print(f"before dropping na {df.count()}")
df.dropna()
print(f"after dropping na {df.count()}")

print(df.dtypes)
print(df.select('date').show(5))
form = 'EEE MMM dd H:m:s z yyyy'
df2 = df.withColumn('date', F.unix_timestamp('date', form).cast('timestamp'))
print(df2.show(5))

df = df2

import matplotlib.pyplot as plt
dates = df.select(F.date_format('date','yyyy-MM-dd').alias('no_timestamp')).groupby('no_timestamp').count().sort(F.col('no_timestamp'))
print(dates.show(dates.count()))
dates.toPandas().plot(kind='line',x='no_timestamp',y='count')

dates.toPandas().plot(kind='bar',x='no_timestamp')

tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
prep_df = tokenizer.transform(df)
cv_prep = CountVectorizer(inputCol="words", outputCol="prep")
cv_model = cv_prep.fit(prep_df)
ready_df = cv_model.transform(prep_df)
# stopWords = [word for word in cv_prep.vocabulary if any(char.isdigit() for char in word)]
# remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords = stopwords)
# prep_df = remover.transform(prep_df)

trainable = ready_df.select('tweet_id', 'prep').rdd.map(lambda x,y: [x,Vectors.fromML(y)]).cache()
print("Trainable")
print(trainable.take(10))
print("take")
model = LDA.train(trainable, k=5, seed=1, optimizer="online")
exit(0)
#Print the topics in the model
topics = model.describeTopics(maxTermsPerTopic = 15)
for x, topic in enumerate(topics):
    print('topic nr: ' + str(x))
    words = topic[0]
    weights = topic[1]
    for n in range(len(words)):
        print(cv_prep.vocabulary[words[n]] + ' ' + str(weights[n]))