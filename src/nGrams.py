

def generateNGrams(df):
    
    clean_reviews_df = df[['line_item_id', 'clean_reviews']]
    
    ngram01 = NGram(n=1, inputCol="clean_reviews", outputCol="ngrams01")
    ngram02 = NGram(n=2, inputCol="clean_reviews", outputCol="ngrams02")
    ngram03 = NGram(n=3, inputCol="clean_reviews", outputCol="ngrams03")

    ngram_pipeline = Pipeline().setStages([ngram01, ngram02, ngram03,])

    ngrams = (ngram_pipeline
              .fit(clean_reviews_df)
              .transform(clean_reviews_df))

    def nGramArray(row):
        return ((row.line_item_id, ) + (row.ngrams01 + row.ngrams02 + row.ngrams03,)) 

    ngrams = (ngrams
              .rdd.map(nGramArray)
              .toDF(['line_item_id', 'ngrams_1to3'])
              .join(ngrams, 'line_item_id', 'left')
              .drop('ngrams01', 'ngrams02', 'ngrams03')
            )
    ngrams = ngrams.drop('clean_reviews')

    df = df.join(ngrams, on = 'line_item_id', how = 'left')
    print('Ngrams generated!')
    return df
