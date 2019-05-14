
def fitTransformLift(df):

    liftDF = df[[[id_var] + ['submitted'] + liftVarList]]

    def likelyhoodByEvent(df, var1, var2):

        window_ul_1 = Window.partitionBy([F.col(var1)])
        df = df.withColumn('var1CountCol', F.count(F.col(var1)).over(window_ul_1))

        window_ul_2 = Window.partitionBy([F.col(var2)])
        df = df.withColumn('var2CountCol', F.count(F.col(var2)).over(window_ul_2))

        window_ul_3 = Window.partitionBy([F.col(var1), F.col(var2)])
        df = df.withColumn('var1var2CountCol', F.count(F.col(var2)).over(window_ul_3))

        v1v2Count = df.select(var1, var2).distinct().count()
        lift_var1var2Col = 'lift_{}_{}'.format(var1, var2)
        df = df.withColumn(lift_var1var2Col,                        
                            (v1v2Count * F.col('var1var2CountCol')) /
                            (F.col('var1CountCol') * F.col('var2CountCol')))

        return df.drop('var1CountCol', 'var2CountCol', 'var1var2CountCol')



    def likelyhoodByAmount(df, var1, var2):
        
        window_ul_1 = Window.partitionBy([F.col(var1)])
        df = df.withColumn('var1AmtCol', F.sum(F.col('submitted')).over(window_ul_1))

        window_ul_2 = Window.partitionBy([F.col(var2)])
        df = df.withColumn('var2AmtCol', F.sum(F.col('submitted')).over(window_ul_2))

        window_ul_3 = Window.partitionBy([F.col(var1), F.col(var2)])
        df = df.withColumn('var1var2AmtCol', F.sum(F.col('submitted')).over(window_ul_3))

        lift_amt_var1var2Col = 'lift_amt_{}_{}'.format(var1, var2)
        df = df.withColumn(lift_amt_var1var2Col,                        
                            F.col('var1var2AmtCol') /
                            (F.col('var1AmtCol') * F.col('var2AmtCol')))

        return df.drop('var1AmtCol', 'var2AmtCol', 'var1var2AmtCol')


    from pyspark.sql.dataframe import DataFrame
    def transform(self, f):
        return f(self)


    DataFrame.transform = transform


    for v1,v2 in cb(liftVarList, 2): 
        liftDF = liftDF.transform(lambda liftDF: likelyhoodByEvent(liftDF,v1,v2))
        liftDF = liftDF.transform(lambda liftDF: likelyhoodByAmount(liftDF,v1,v2))
    liftDF = liftDF.rdd.toDF()


    liftDF = liftDF.select(['line_item_id'] + [e for e in liftDF.columns if e.startswith('lift')])
    df = df.join(liftDF, on = 'line_item_id', how = 'left')
    
    print('Lift features created!')
    return df

