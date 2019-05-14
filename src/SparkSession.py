import pyspark
from pyspark import SparkConf

from pyspark.sql import (
    SparkSession,
    functions as F,
    types as T,
    Window
)
spark = (SparkSession.builder
         .appName("Chubb - Data Preparation")
         .config('spark.master', 'yarn')
         .config('spark.sql.cbo.enabled', True)
         .config('spark.sql.cbo.joinReorder.enabled', True)
         .config("spark.executor.instances",10)
         .config("spark.executor.cores",5)
         .config("spark.executor.memory", "8g")
         .config('spark.driver.memory', '4g')
         .config('spark.yarn.executor.memoryOverhead', '4g')
         .config('spark.yarn.queue', 'root.default')
         .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:1.2.3")
         .config("spark.port.maxRetries", 100)
         .config('spark.default.parallelism', 100)
         .config('spark.kryoserializer.buffer.max', '512m')
         .config('spark.dynamicAllocation.enabled', True)
         .config('spark.shuffle.service.enabled', True)
         .getOrCreate()
         )

from pyspark.ml.feature import (Tokenizer, NGram, RegexTokenizer as smlRegexTokenizer,
                                StopWordsRemover as sml_StopWordsRemover,CountVectorizer, 
                                CountVectorizerModel, IDF,IDFModel, Word2Vec, StringIndexer, 
                                VectorIndexer, IndexToString, QuantileDiscretizer, VectorAssembler,
                               Word2VecModel
                               )

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.ml.classification import (RandomForestClassifier,
                                       RandomForestClassificationModel)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                  MulticlassClassificationEvaluator)
from pyspark.ml.tuning import (CrossValidator, CrossValidatorModel,
                               ParamGridBuilder)
from pyspark.ml.linalg import DenseVector, SparseVector, Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline, PipelineModel

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

import os
import fnmatch
import pandas as pd
import numpy as np
import re
from copy import deepcopy
pd.set_option('display.max_columns', 999)
import logging
from time import time
from datetime import datetime
from itertools import combinations as cb


exec(open('../src/variable_utilities_pyspark.py').read(), globals())
exec(open('../src/class_utilities_pyspark.py').read(), globals())