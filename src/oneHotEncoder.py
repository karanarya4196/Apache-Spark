

def oneHotEncoding(df):
    

    df = (Pipeline(stages =
         [StringIndexer()
        .setInputCol(col)
        .setOutputCol(col + '_catIndex')
        .setHandleInvalid('skip') # check what does skip do
        .fit(df) for col in catVars])
        .fit(df).transform(df))


    df = (Pipeline(stages =
                        [(OneHotEncoder()
                          .setInputCol(col + '_catIndex')
                          .setOutputCol(col + '_catIndex' + '_oneHotVec'))
                         for col in catVars])
                      .fit(df).transform(df))
    
    print('One hot encoding completed!')
    return df

