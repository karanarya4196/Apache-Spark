word2VecModel = Word2VecModel.load('{}/modeloriginal'.format(clientName))

train_df = word2VecModel.transform(train_df)
dev_df = word2VecModel.transform(dev_df)



def word2VecStacking(train_df, dev_df):
    
    labelIndexer = (StringIndexer()
                  .setInputCol('reviewer_adjusted')
                  .setOutputCol('indexedLabel')
                  .fit(train_df) )
    
    rf = (RandomForestClassifier()
    .setNumTrees(500)
    .setMinInstancesPerNode(50)
    .setMaxDepth(2)
    .setMaxBins(2)
    .setFeaturesCol('W2V')
    .setLabelCol('indexedLabel')
    .setRawPredictionCol('W2V_RawPred')
    .setPredictionCol('W2V_Pred')
    .setProbabilityCol('W2V_Probs')
    .setSeed(452310) )
    
    labelConverter = (IndexToString()
                .setInputCol('W2V_Pred')
                .setOutputCol('predictedLabel')
                .setLabels(labelIndexer.labels)  )
    


    pipeline_rf = (Pipeline()
                 .setStages([labelIndexer,    
                             rf,               
                             labelConverter])) 

    rf_pipe_model = pipeline_rf.fit(train_df)
    train_df = rf_pipe_model.transform(train_df)
    dev_df = rf_pipe_model.transform(dev_df)

    def extract(row):
        return (row.line_item_id, ) + tuple(row.W2V_Probs.toArray().tolist())

    prob_train_df = (train_df
                       .select('line_item_id','W2V_Probs')
                       .rdd.map(extract)
                       .toDF(['line_item_id', 'W2V_prob_NotAdjusted','W2V_prob_Adjusted'])
                       .drop('W2V_prob_NotAdjusted') )

    prob_dev_df = (dev_df
                    .select('line_item_id','W2V_Probs')
                    .rdd.map(extract)
                    .toDF(['line_item_id', 'W2V_prob_NotAdjusted','W2V_prob_Adjusted'])
                    .drop('W2V_prob_NotAdjusted') )
    train_df = train_df.join(prob_train_df, on = 'line_item_id', how = 'left')
    dev_df = dev_df.join(prob_dev_df, on = 'line_item_id', how = 'left')

    dropCols = ['indexedLabel', 'W2V_RawPred', 'W2V_Probs', 'W2V_Pred', 'predictedLabel']
    for col in dropCols:
        train_df = train_df.drop(col)
        dev_df = dev_df.drop(col)

    print('Word2Vec stacking completed!')
    return train_df, dev_df

