
def fitTransformOOP(df):

    OOP_features = ['avg_' +target+ '_' +source for source in sourceList for target in targetList]
    OOP_df = df[[['line_item_id'] + sourceList + targetList]]
    def avgTargetAcrossSource(OOP_df, target, source):
        window = Window.partitionBy(source)
        OOP_df = OOP_df.withColumn('avg_%s_%s'%(target, source), 
                             F.when(F.avg(F.col(target)).over(window) == 0, 
                                    F.col(target)/F.avg(F.col(target)).over(window))
                              .otherwise(F.col(target)))
        return OOP_df

    from pyspark.sql.dataframe import DataFrame
    def transform(self, f):
        return f(self)
    DataFrame.transform = transform
    
    for source in sourceList:
        for target in targetList:
            OOP_df = OOP_df.transform(lambda OOP_df: avgTargetAcrossSource(OOP_df, target, source))
    
    OOP_df = OOP_df.rdd.toDF()
    OOP_df = OOP_df.select(['line_item_id'] + [e for e in OOP_df.columns if e.startswith('avg_')])
    
    df = df.join(OOP_df, on = 'line_item_id', how = 'left')
    df = df.na.fill(0, subset = OOP_features)
    print('OOP features generated!')
    return df

