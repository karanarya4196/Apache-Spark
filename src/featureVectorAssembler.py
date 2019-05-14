def featureVectorAssembler(df):
    
    excludeList = ['line_item_id', 'task_code', 'role_name', 'activity_code', 'reviewer_adjusted', 'category_var']
    xCols = [c for c in df.columns if c not in excludeList]
    print('Number of input_X columns:', len(xCols))
    assembler = (VectorAssembler()
              .setInputCols(xCols)
              .setOutputCol("features"))
    df_1 = assembler.transform(df).rdd.toDF().cache()
    print('Feature columns assembled!')
    return df_1
