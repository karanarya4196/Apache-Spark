def convertDataTypes(df):
    
    for col_name, data_type in schema.items():
        
        if data_type == 'str':
            df = df.withColumn(col_name, df[col_name].cast(T.StringType()))
        elif data_type == 'int':
            df = df.withColumn(col_name, df[col_name].cast(T.IntegerType()))
        elif data_type == 'float':
            df = df.withColumn(col_name, df[col_name].cast(T.DoubleType()))
        elif data_type == 'date':
            df = df.withColumn(col_name, F.unix_timestamp(col_name, "yyyy-MM-dd HH:mm:ss.000") .cast(T.TimestampType()))
            df = df.withColumn(col_name, F.to_date(F.col(col_name)))
        else:
            raise ValueError('Unexpected dtype "%s" specified for column "%s"' %(data_type, col_name))
    print('Data types converted!')
    return df