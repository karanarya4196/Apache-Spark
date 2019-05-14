def readCSVData(filePath):
    df = spark.read.option('inferSchema', 'True').option('header', 'True').csv(filePath)
    print('Shape of dataframe is:', df.count(), len(df.columns))
    print('Columns of dataframe are:', df.columns)
    return df


def readParquetData(filePath):
    df = spark.read.option('inferSchema', 'True').option('header', 'True').parquet(filePath)
    print('Shape of dataframe is:', df.count(), len(df.columns))
    print('Columns of dataframe are:', df.columns)
    return df