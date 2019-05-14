
def cleanDescription(df):

    data = df[['line_item_id','description']].na.fill('Null_Value')
    
    documentAssembler = (DocumentAssembler() 
      .setInputCol("description") 
      .setOutputCol("assembled_document") 
      .setIdCol("line_item_id")
      )

    tokenizer = (RegexTokenizer() 
    .setInputCols(["assembled_document"])
    .setOutputCol("token") 
    .setPattern("\w+")
    )


    normalizer = (Normalizer() 
      .setInputCols(["token"]) 
      .setOutputCol("normalized")
      .setPattern("[^A-Za-z]")
      )

    stemmer = (Stemmer() 
       .setInputCols(["normalized"]) 
       .setOutputCol("stem")
       )

    
    finisher = (Finisher() 
    .setInputCols(["stem"])
    .setOutputCols(["stem_tokens"])
    .setAnnotationSplitSymbol(', ')
    .setValueSplitSymbol('|')
    .setCleanAnnotations(True)
    .setIncludeKeys(False)
    .setOutputAsArray(True)
    )

    remover = sml_StopWordsRemover(inputCol='stem_tokens', 
                              outputCol="clean_reviews",
                              stopWords=stopwordList)

    nlp_pipeline = Pipeline() \
      .setStages([
        documentAssembler, 
        tokenizer,
        normalizer, 
        stemmer,
        finisher,
        remover
      ])

    clean_description_stages = list(enumerate(nlp_pipeline.getStages()))
    nlp_pl_model = nlp_pipeline.fit(data)
    clean_description = (nlp_pl_model.transform(data))      
    
    clean_description = clean_description.drop('description')
    
    df = df.join(clean_description, on = 'line_item_id', how = 'left')
    print('Cleaning of description completed!')
    return df

def createDescriptionString(df):
    df = df.withColumn("clean_reviews_string", F.concat_ws(" ", "clean_reviews"))
    print('Clean description sentences created!')
    return df


def computeCleanDescriptionLength(df):

    df = df.withColumn('wordCount', F.size(F.col('clean_reviews')))
    print('Length of clean description computed!')
    return df


