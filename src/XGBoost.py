df = spark.read.parquet("{}/data/allFeaturesXGB_df.parquet".format(clientName))
train_df = spark.read.parquet("{}/data/allFeaturesXGB_train_df.parquet".format(clientName))
dev_df = spark.read.parquet("{}/data/allFeaturesXGB_dev_df.parquet".format(clientName))


spark.sparkContext.addPyFile("sparkxgb.zip")


from sparkxgb import XGBoostEstimator, XGBoostClassificationModel, XGBoostPipeline, XGBoostPipelineModel



xgboost = XGBoostEstimator(
    
    # General Params
    nworkers = 4, 
    silent = 0, 
    
    # Column Params
    featuresCol = "features", 
    labelCol = "indexedLabel", 
    predictionCol = "prediction", 
    weightCol = "weight", 
    baseMarginCol = "baseMargin", 
    
    # Booster Params
    booster = "gbtree", 
    base_score = 0.5, 
    objective = "binary:logistic", 
    eval_metric = "auc", 
    num_class = 2, 
    num_round = 1000, 
    seed = 12345,
    
    # Tree Booster Params
    eta = 0.01, 
    max_depth = 3, 
    min_child_weight=1.0, 
    max_delta_step=0.0, 
    subsample = 0.6,
    colsample_bytree = 0.3, 
    colsample_bylevel=1.0, 
    reg_lambda = 0.8, 
    alpha = 0.4 
)
 
Y_indexer = StringIndexer()\
            .setInputCol("reviewer_adjusted")\
            .setOutputCol("label")\
            .fit(train_df)
            
labelConverter = IndexToString()\
                .setInputCol("prediction")\
                .setOutputCol("pred_Y")\
                .setLabels(Y_indexer.labels)



xgbPipeline = Pipeline().setStages([xgboost, labelConverter])
xgbPipelineModel = xgbPipeline.fit(train_df)
dev_prediction = xgbPipelineModel.transform(dev_df)



xgbPipelineModel.stages[0].write().overwrite().save('{}/output/xgbModel'.format(clientName))
dev_prediction.write.parquet("{}/output/allXGB_DevPrediction.parquet".format(clientName), mode = 'overwrite')

allFeaturesPath = '{}/output/allXGB_DevPrediction.parquet'.format(clientName)
allFeaturesPred = readParquetData(allFeaturesPath)


window1 = Window.orderBy('adjustedProbScore').rowsBetween(Window.unboundedPreceding, Window.currentRow)
window2 = Window.orderBy('adjustedProbScore').rowsBetween(Window.currentRow, Window.unboundedFollowing)

allFeaturesPred = allFeaturesPred\
    .withColumn('int', F.lit(1))\
    .withColumn('PredP_AtProb', F.sum('int').over(window2))\
    .drop('int')\
    .withColumn('P', F.lit(allFeaturesPred.where(F.col('reviewer_adjusted') == 1).count()))\
    .withColumn('N', F.lit(allFeaturesPred.where(F.col('reviewer_adjusted') == 0).count()))\
    .withColumn('TP', F.sum('reviewer_adjusted').over(window2))\
    .withColumn('FP', F.col('PredP_AtProb') - F.col('TP'))\
    .withColumn('TPR', F.col('TP')/F.col('P'))\
    .withColumn('FPR', F.col('FP')/F.col('N'))\
    .withColumn('1-FPR', 1 - F.col('FPR'))\
    .withColumn('TPR-(1-FPR)', F.abs(F.col('TPR') - F.col('1-FPR')))



def plotROC(allFeaturesPred):
    
    TPR_range = allFeaturesPred.select("TPR").rdd.flatMap(lambda x: x).collect()
    FPR_range = allFeaturesPred.select("FPR").rdd.flatMap(lambda x: x).collect()
    p = figure(title = "ROC curve", x_axis_label = 'FPR', y_axis_label = 'TPR',
              x_range=(0, 1), y_range=(0, 1))
    p.line(FPR_range, TPR_range, legend = "XGB", line_width = 2)

    html_plot_roc = file_html(p, CDN, "ROC")
    with open('../output/ROC_Curve.html', "w") as file:
        file.write(html_plot_roc)

    HTML(html_plot_roc)
    
plotROC(allFeaturesPred)

cutOff = 0.3
allFeaturesPred = allFeaturesPred.withColumn('thresholdPred', F.when(F.col('adjustedProbScore') > cutOff, 1).otherwise(0))
getModelEvaluation(allFeaturesPred, true_Y = 'reviewer_adjusted', pred_Y = 'thresholdPred')