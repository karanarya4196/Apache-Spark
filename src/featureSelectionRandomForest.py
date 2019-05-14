

def fitFeatureSelection(df):

    rf = (RandomForestClassifier()
          .setNumTrees(1000)
          .setMinInstancesPerNode(50)
          .setMaxDepth(30)
          .setMaxBins(30)
          .setFeaturesCol('features')
          .setLabelCol('indexedLabel')
          .setRawPredictionCol('rf_RawPred')
          .setPredictionCol('rf_Pred')
          .setProbabilityCol('rf_Probs')
          .setSeed(452310) )

    labelConverter = (IndexToString()
                      .setInputCol("rf_Pred")
                      .setOutputCol("predictedLabel")
                      .setLabels(labelIndexer.labels))

    rf_final_pipeline = (Pipeline(stages=[
                                    rf,
                                    labelConverter
                                    ]))

    rf_final_pipeline_model = rf_final_pipeline.fit(df)

    print('Fit train data to feature selection model!')
    return rf_final_pipeline_model


def transformFeatureSelection(df, rf_final_pipeline_model):
    df = rf_final_pipeline_model.transform(df)

    labelIndexer.labels
    target = y_var
    predicted = 'predictedLabel'

    evaluator = (BinaryClassificationEvaluator()
             .setLabelCol('indexedLabel')
             .setRawPredictionCol('rf_RawPred')
             .setMetricName('areaUnderROC')
            ) 

    roc = evaluator.evaluate(df)
    print('ROC = ' + str(roc))


    evaluator2 = (MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("rf_Pred")
                .setMetricName("weightedRecall")) 

    weightedRecall = evaluator2.evaluate(df)
    print('Weighted Recall = ' + str(weightedRecall))

    evaluator3 = (MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("rf_Pred")
                .setMetricName("accuracy")) 

    accuracy = evaluator3.evaluate(df)
    print('Accuracy = ' + str(accuracy))


    FeatImporCol = [(i, float(j),) for i,j in list(enumerate(rf_final_pipeline_model.stages[0].featureImportances))]
    print(len(rf_final_pipeline_model.stages[0].featureImportances))
    print('Feature selection transformation completed!')
    return FeatImporCol, df

