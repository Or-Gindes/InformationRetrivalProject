# Information Retrieval Project
Author: Or Gindes\
Team: CC\
Submission Date: 15-Jan-2023

Code Structure and Flow -
* inverted_index_gcp / inverted_index_colab - 
  * Used to generate and load InvertedIndices which allows to prepare and retrieve 
  document and query information for ranking calculations.
  * The code is mostly used a provided for assignment 3 but altered to allow .bin files
  to be saved in specified folders to solve the issue of overwriting when running multiple 
  indices in one session.
* createTrainInvertedIndices - this jupyter notebook was used to generate the indices with
  including some adjustments for each iteration such as running only on training vocabulary
  (i.e. words found in train_queries), stemming or lemmatization.
* search_frontend - Implementation of all search functions
  * search_body - based on tfidf and cosine similarity metric
  * search_title & search_anchor - based on binary match function. 
  Return articles which have the highest number of matching terms
  * get_pageView and get_pageRank - return relevant values for document ids provided
  * search - the final search function
    * Final variation uses a weighted cosine similarity for body, title and anchors
    * Multiple weights allocation were tested before final ones were chosen
    * bm25 similarity score was also tested with multiple weights and k,b values but results were not improved
    * likewise, stemming and lemmatization have also not yielded improved results
      (stemming was generated for body, title and anchor while lemmatization was only tested on text portion 
    due to it being the largest and therefor the one most likely to benefit from the process)
    * gensim Word2Vec model was also tested for query expansion
      * The query was expended using top2/top3 most similar terms
      * response time increased but MAP@40 didn't improve with these changes
      * Score was augmented with the similarity metric (original tokens didn't have their score changed)
      but this too failed to yield improved results.


Important Links -
* GCP Bucket - https://console.cloud.google.com/storage/browser/201640042_project
  * TextIndex - https://console.cloud.google.com/storage/browser/201640042_project/full_body_index
  * TitleIndex - https://console.cloud.google.com/storage/browser/201640042_project/full_title_index
  * AnchorIndex - https://console.cloud.google.com/storage/browser/201640042_project/full_anchor_index
* Git Repository - https://github.com/Or-Gindes/InformationRetrivalProject