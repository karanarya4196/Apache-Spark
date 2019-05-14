
def fit_corpus2TFIDF(df):
    
    ngrams = df[['line_item_id', 'clean_reviews', 'ngrams_1to3']]
    
    tf_vectorizer = (CountVectorizer().setInputCol('ngrams_1to3')
                    .setOutputCol('ng1to3_tf')
                    .setBinary(False)
                    .setMinDF(10)
                    .setMinTF(1)
                    .setVocabSize(2000))
    
    idf_vectorizer = IDF(inputCol="ng1to3_tf", outputCol="ng1to3_tfidf", 
                      minDocFreq = 10)
    
    tfidf_pipeline = Pipeline().setStages([tf_vectorizer, idf_vectorizer])

    tfidf_model_pipeline = tfidf_pipeline.fit(ngrams)

    ngrams = ngrams.drop('clean_reviews')
    print('Fit train data to TFIDF model!')
    return ngrams, tfidf_model_pipeline


def saveTfWords(tfidf_model_pipeline):
        
    tf_vocab = [x for x in list(enumerate(tfidf_model_pipeline.stages[0].vocabulary))]
    
    (spark.createDataFrame(tf_vocab, ['word_num','ng1to3_vocabWords'])
    .write.parquet("{}/output/ng1to3_vocabWords.parquet".format(clientName), mode = 'overwrite'))
    print('Saved TF Vocabulary!')

def transform_corpus2TFIDF(df, ngrams, tfidf_model_pipeline):

    tf_vocab_list = tfidf_model_pipeline.stages[0].vocabulary
    
    def TF_VocabWords(row):
        return ((row.line_item_id, )
                + ([tf_vocab_list[i] for i in 
                    [i for i,j in 
                    enumerate(row.ngram_tf.toArray().tolist()) 
                    if j > 0]],))

    
    tf_vocab_ng1to3 = (tfidf_model_pipeline.stages[0].transform(ngrams)
                    .withColumnRenamed('ng1to3_tf', 'ngram_tf')
                    .select('line_item_id','ngram_tf')
                    .rdd.map(TF_VocabWords)
                    .toDF(['line_item_id', 'tf_vocab_ng1to3']))
        
    ngrams_tfidf = tfidf_model_pipeline.transform(ngrams)
    ngrams_tfidf = ngrams_tfidf.drop('ngrams_1to3')
    
    df = df.join(ngrams_tfidf, on = 'line_item_id', how = 'left')
    df = df.join(tf_vocab_ng1to3, on = 'line_item_id', how = 'left')
    print('TFIDF transform completed!')
    return tf_vocab_list, tf_vocab_ng1to3, ngrams_tfidf, df


