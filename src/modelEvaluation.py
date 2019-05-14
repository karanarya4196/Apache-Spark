

def getModelEvaluation(df, true_Y, pred_Y):
    
    df = df.withColumn(pred_Y, df[pred_Y].cast(T.DoubleType()))
    confusionMatrix = df.groupBy(true_Y, pred_Y).count()

    
    TP = float(confusionMatrix.filter((F.col(pred_Y) == 1) & (F.col(true_Y) == 1)).select('count').collect()[0][0])
    FP = float(confusionMatrix.filter((F.col(pred_Y) == 1) & (F.col(true_Y) == 0)).select('count').collect()[0][0])
    FN = float(confusionMatrix.filter((F.col(pred_Y) == 0) & (F.col(true_Y) == 1)).select('count').collect()[0][0])
    TN = float(confusionMatrix.filter((F.col(pred_Y) == 0) & (F.col(true_Y) == 0)).select('count').collect()[0][0])
    
    
    P = float(TP + FN)
    N = float(FP + TN)
    TPR = float(TP)/P 
    TNR = float(TN)/N 

    confusionMatrix.show()
    precision = float(TP)/(TP + FP)*100
    print('Precision = ' + str(precision))
    recall = float(TP)/(TP + FN)*100
    print('Recall, Sensitivity = ' + str(recall))
    specificity = TNR
    print('Specificity = ' + str(specificity))
    roc_evaluator = (BinaryClassificationEvaluator()
                 .setLabelCol(true_Y)
                 .setRawPredictionCol('probabilities')
                 .setMetricName('areaUnderROC')
                ) 
    roc = roc_evaluator.evaluate(df)
    print('ROC = ' + str(roc))
    
    
    f1_evaluator = (MulticlassClassificationEvaluator()
            .setLabelCol(true_Y)
            .setPredictionCol(pred_Y)
            .setMetricName("f1")) 
    f1 = f1_evaluator.evaluate(df)
    print('F1 Score = ' + str(f1))

    
    accuracy_evaluator = (MulticlassClassificationEvaluator()
                .setLabelCol(true_Y)
                .setPredictionCol(pred_Y)
                .setMetricName("accuracy")) 

    accuracy = accuracy_evaluator.evaluate(df)
    print('Accuracy = ' + str(accuracy))
