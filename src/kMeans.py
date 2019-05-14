
def fitKMeansClustering(df):
 
    kmeans = (KMeans()
            .setFeaturesCol("ng1to3_tfidf")
            .setPredictionCol("ng1to3_kMeans")
            .setK(50)
            .setSeed(1234)
            .setInitMode('k-means||') 
            .setInitSteps(2)         
            .setMaxIter(500)
           )

    tfidf_kMeans_model = kmeans.fit(df)

    tfidf_kMeans_model.clusterCenters()
    tfidf_kMeans_model.computeCost(df)
    tfidf_kMeans_model.hasSummary
    
    print('Fit TFIDF vectors of train data to kMeans model!')
    return tfidf_kMeans_model

def transformKMeansClustering(df, tfidf_kMeans_model):

    tfidf_kMeans = tfidf_kMeans_model.transform(df).select(['line_item_id', 'ng1to3_kMeans'])
    
    df = df.join(tfidf_kMeans, on = 'line_item_id', how = 'left')
    print('TFIDF kMeans stacking transformation completed!')
    return df
