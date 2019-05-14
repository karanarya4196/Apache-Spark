def createMissingIndicator(df, input_cols = missing_indicator_cols):
    
    temp_missing_df = df
    for col in input_cols:
        temp_missing_df = temp_missing_df.withColumn(col+'_missing', (F.isnull(F.col(col))).cast(T.IntegerType()))
    print('Missing indicators created!')
    return temp_missing_df
