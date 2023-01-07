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

Action items -
* MIN- REQUIREMENTS
  * ~~Generate Body / Title InvertedIndex on query vocabulary - GCP~~ - **DONE**
  * Repeat with stemming - Generated, need to test
  * Repeat with lemmatization - Need to generate & test
  * Test query return time & MAP@40 - colab & gcp
    * Make sure search_frontend runs in gcp
    * ~~SearchBody - only basic index~~ - **DONE**
    * ~~SearchTitle - only basic index~~ - **DONE**
    * Weighted Search Function - ~~basic~~, stemming, lemmatization
  * Generate Full Body and Title InvertedIndex based on best results (stemming/lemmatization)
  * Submission Report
    * Qualitative evaluation of the top 10 results for one query where your engine
    performed really well and one query where your engine did not perform well
* Extra Credit
  * ~~Anchor text Index + Anchor search function~~ - **DONE**
  * (HW1) PageViews Index + search - Generated Data in bucket, need to write function
  * (HW4) PageRank Index + search
  * Test Search function with Anchor / PageViews / PageRank weights / BM25
  * Word2Vec Query expansion