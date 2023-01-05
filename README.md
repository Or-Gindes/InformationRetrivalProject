# InformationRetrivalProject

Min Requirements -
* Functional and testable search engine -
  * OR - prepare mini-search engine in colab
  * Yonatan - prepare bucket & cluster in gcp and deploy mini-engine
* Efficiency - No query takes longer than 35 seconds to process
* Quality - MAP@40 > 0.25
* Report - 
  * A link to a Google Storage Bucket, where all indexing data you calculated resides
and is publicly accessible. Instructions for making your bucket public are here. 
  * List all index files with human-readable sizes, pasted in an appendix at the end of your report. 
  This does not count towards the page limit. You can use the command `gsutil du -ch gs://BUCKET_NAME` to generate this listing or `du -ch
LOCAL_DIR` if you store index data locally. 
  * A description of key experiments you ran, how you evaluated them, and the key findings/takeaways. 
  * A graph showing the engine performance for each major version of your implementation. 
  * A graph showing the engine's average retrieval time for each major version of your implementation.
  * Qualitative evaluation of the top 10 results for one query where your engine
  performed really well and one query where your engine did not perform well.
  Describe what worked well and what didn't work so well. What is the dominant
  factor behind the poor result? What can be done about it?


Things we plan to try -

Inverted index - 
* Remove English stopwords + corpus stopwords
* Stemming + stopwords removal
* Lemmatization
* word2vec? Query expansion

Search functions -
* cosine similarity tf-idf
* binary ranking using titles
* binary ranking using anchor text
* BM25
* PageRank
* Ranking by page views
* Deep learning