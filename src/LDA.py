def fitLDA(df): 

    lda = LDA(featuresCol = 'ng1to3_tfidf',
            topicDistributionCol = 'lda',
            k=50, 
            maxIter=10, 
            optimizeDocConcentration = True,
            optimizer = 'online',
            learningOffset = 50, 
            seed = 1234,
            subsamplingRate = 0.05,
            keepLastCheckpoint = True,
            learningDecay = 0.51
            )

    tfidf_lda_model = lda.fit(df)

    tfidf_lda_model.isDistributed()
    tfidf_lda_model.vocabSize()
    tfidf_lda_model.topicsMatrix()
    tfidf_lda_model.estimatedDocConcentration()

    ldaTopics = tfidf_lda_model.describeTopics()
    print('Fit TFIDF vectors of train data to LDA model!')
    return tfidf_lda_model


def transformLDA(df, tfidf_lda_model): 

    tfidf_lda = tfidf_lda_model.transform(df).select(['line_item_id', 'lda'])    
    df = df.join(tfidf_lda, on = 'line_item_id', how = 'left')
    print('LDA topic modeling completed!')
    return df

def fitLDA_kMeans(df):
 
    KMeans().getTol()

    kmeans = (KMeans()
            .setFeaturesCol("lda")
            .setPredictionCol("lda_kMeans")
            .setK(50)
            .setSeed(1234)
            .setInitMode('k-means||')
            .setInitSteps(2)     
            .setMaxIter(500)
           )

    lda_kMeans_model = kmeans.fit(df)
    
    print('Fit LDA topics of train data to kMeans model!')
    return lda_kMeans_model

def transformLDA_kMeans(df, lda_kMeans_model):

    lda_kMeans = lda_kMeans_model.transform(df).select(['line_item_id', 'lda_kMeans'])    
    df = df.join(lda_kMeans, on = 'line_item_id', how = 'left')
    print('kMeans clustering of LDA completed!')
    return df

