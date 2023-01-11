from flask import Flask, request, jsonify
import pandas as pd
from collections import defaultdict, Counter
import re
import nltk
from nltk.corpus import stopwords
import pickle
import numpy as np
from math import sqrt, pow
import os
from nltk.stem import WordNetLemmatizer, PorterStemmer
# import gensim.downloader as api
# model = api.load('glove-wiki-gigaword-200')
from inverted_index_gcp import *
# from inverted_index_colab import *

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

BODY_INDEX = 'full_body_index'
TITLE_INDEX = 'full_title_index'
ANCHOR_INDEX = 'full_anchor_index'
index_text = InvertedIndex().read_index(os.path.join(os.getcwd(), BODY_INDEX), BODY_INDEX)
index_title = InvertedIndex().read_index(os.path.join(os.getcwd(), TITLE_INDEX), TITLE_INDEX)
index_anchor = InvertedIndex().read_index(os.path.join(os.getcwd(), ANCHOR_INDEX), ANCHOR_INDEX)

index_dict = {BODY_INDEX: index_text, TITLE_INDEX: index_title, ANCHOR_INDEX: index_anchor}
AVG_DOC_LEN = {index_name: sum([data[1] for doc_id, data in index.doc_data.items()]) / index._N for index_name, index in index_dict.items()}

with open("pageviews-202108-user.pkl", 'rb') as f:
    PAGE_VIEWS = defaultdict(int,pickle.loads(f.read()))
    
with open("pagerank_org.pkl", 'rb') as f:
    PAGERANK = defaultdict(int,pickle.loads(f.read()))
    
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]
ALL_STOPWORDS = stopwords_frozen.union(corpus_stopwords)


def tokenize(text, stem=False, lemm=False):
    """
    This function turns text into a list of tokens. Moreover, it filters stopwords.
    Parameters:
        lemm: lemmatize tokens
        stem: stem tokens
        text: string , represting the text to tokenize.
    Returns:
         list of tokens (e.g., list of tokens).
    """

    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in ALL_STOPWORDS]
    if stem:
        return [stemmer.stem(tok) for tok in list_of_tokens]
    if lemm:
        return [lemmatizer.lemmatize(tok) for tok in list_of_tokens]
    return list_of_tokens


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # Tokenize query
    tokenized_query = tokenize(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    
#     lemm_tokenized_query = tokenize(query, lemm=True)
#     query expansion - removed due to queries taking too long to return and results weren't vastly improved
#     word2vec_sim_score = defaultdict(float)
#     for term in tokenized_query:
#         word2vec_sim_score[term] = 1.0
#         try:
#             similar_terms = model.most_similar(term, topn=2)
#         except KeyError:
#             continue
#         for sim_term, similarity in similar_terms:
#             word2vec_sim_score[sim_term] = max(similarity, word2vec_sim_score[sim_term])
#     tokenized_query = list(word2vec_sim_score.keys())
    
    title_weight, body_weight, anchor_weight = 0.35, 0.50, 0.15 # TODO: dynamic weights by token length
    merged_score = defaultdict(float)
    sorted_score_body = cosin_similarity_score(tokenized_query, index_text, BODY_INDEX)
    sorted_score_title = cosin_similarity_score(tokenized_query, index_title, TITLE_INDEX)
    sorted_score_anchor = cosin_similarity_score(tokenized_query, index_anchor, ANCHOR_INDEX)
    
    for doc_id, sim_score in sorted_score_body.items():
        merged_score[doc_id] += sim_score * body_weight
    for doc_id, sim_score in sorted_score_title.items():
        merged_score[doc_id] += sim_score * title_weight
    for doc_id, sim_score in sorted_score_anchor.items():
        merged_score[doc_id] += sim_score * anchor_weight

    term_page_ranking = {}
    term_page_views = {}
    for doc_id in merged_score.keys():
        term_page_ranking[doc_id] = PAGERANK[doc_id]
        term_page_views[doc_id] = PAGE_VIEWS[doc_id]

    pagerank = {key: rank for rank, key in enumerate(sorted(term_page_ranking, key=term_page_ranking.get), 1)}
    pageviews = {key: rank for rank, key in enumerate(sorted(term_page_views, key=term_page_views.get), 1)}

    adjusted_scores = {k: v * (pagerank[k] + pageviews[k]) for k, v in merged_score.items()}
    sorted_adjusted_scores = {k: v for k, v in sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True)}

    # add top 100 docs to result
    for i, doc_id in enumerate(sorted_adjusted_scores.keys()):
        if i == 100:
            break
        res.append((doc_id, index_text.doc2title[doc_id]))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # Tokenize query
    tokenized_query = tokenize(query)
    if len(tokenized_query) == 0:
        return jsonify(res)

    sorted_cosin_sim = cosin_similarity_score(tokenized_query, index_text, BODY_INDEX)
    # add top 100 docs to result
    for i, doc_id in enumerate(sorted_cosin_sim.keys()):
        if i == 100:
            break
        res.append((doc_id, index_text.doc2title[doc_id]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # Tokenize query
    tokenized_query = tokenize(query)
    if len(tokenized_query) == 0:
        return jsonify(res)

    match_counter = defaultdict(int)
    # iterate DISTINCT query tokens and for each doc that has that word increase counter by 1
    for term in set(tokenized_query):
        pls = index_title.read_posting_list(term, TITLE_INDEX)
        for doc_id, _ in pls:
            match_counter[doc_id] += 1

    sorted_match_counter = {k: v for k, v in sorted(match_counter.items(), key=lambda item: item[1], reverse=True)}
    for doc_id, match_count in sorted_match_counter.items():
        res.append((doc_id, index_title.doc2title[doc_id]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).
        Test this by navigating to a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # Tokenize query
    tokenized_query = tokenize(query)
    if len(tokenized_query) == 0:
        return jsonify(res)

    match_counter = defaultdict(int)
    # iterate DISTINCT query tokens and for each doc that has that word increase counter by 1
    for term in set(tokenized_query):
        pls = index_anchor.read_posting_list(term)
        for doc_id, _ in pls:
            match_counter[doc_id] += 1

    sorted_match_counter = {k: v for k, v in sorted(match_counter.items(), key=lambda item: item[1], reverse=True)}
    for doc_id, match_count in sorted_match_counter.items():
        res.append((doc_id, index_anchor.doc2title[doc_id]))

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res.extend([PAGERANK[doc_id] for doc_id in wiki_ids])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res.extend([PAGE_VIEWS[doc_id] for doc_id in wiki_ids])
    # END SOLUTION
    return jsonify(res)


def cosin_similarity_score(tokenized_query, index, index_folder="."):
    """
    Support function to calculate cosin similarity
    :param index: InvertedIndex
    :param tokenized_query: list of query tokens post-processing
    :return: sorted dictionary of doc_id: cosin_similarity_score
    """
    cosine_sim_numerator = defaultdict(float)
    query_len = len(tokenized_query)
    tf_query = Counter(tokenized_query)
    query_norm = sum([pow((tf_term / query_len) * index.get_idf(term), 2) for term, tf_term in tf_query.items()])
    for term, count in tf_query.items():
        pls = index.read_posting_list(term, index_folder)
        if pls is None:
            pls = []
        term_idf = index.get_idf(term)
        # normalized query tfidf
        query_tfidf = count / query_len * term_idf
        for doc_id, doc_tf in pls:
            doc_len = index.doc_data[doc_id][1]
            # normalized document tfidf
            doc_tfidf = doc_tf / doc_len * term_idf
            cosine_sim_numerator[doc_id] += doc_tfidf * query_tfidf

    # vector length of each doc is calculated at index creation
    cosine_sim = {doc_id: numerator / sqrt(index.doc_data[doc_id][0] * query_norm) for doc_id, numerator in
                  cosine_sim_numerator.items()}
    sorted_cosin_sim = {k: v for k, v in sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)}
    return sorted_cosin_sim


def bm25_score(tokenized_query, index, index_folder=".", b=0.75, k1=1.5, k3=1.5):
    """
    Support function to calculate bm25 score
    :param index: InvertedIndex
    :param tokenized_query: list of query tokens post-processing
    :return: sorted dictionary of doc_id: bm25_similarity_score
    """
    score = defaultdict(float)
    query_len = len(tokenized_query)
    tf_query = Counter(tokenized_query)
    avg_dl = AVG_DOC_LEN[index_folder]
    for term, count in tf_query.items():
        pls = index.read_posting_list(term, index_folder)
        if pls is None:
            pls = []
        term_idf = index.get_idf(term)
        # normalized query tfidf
        query_tf = count / query_len
        for doc_id, doc_tf in pls:
            doc_len = index.doc_data[doc_id][1]
            numerator = ((k1 + 1) * doc_tf / doc_len) * term_idf * ((k3 + 1) * query_tf)
            denumerator = (k1 * (1 - b + b * doc_len / avg_dl) + doc_tf / doc_len) * (k3 + query_tf)
            score[doc_id] += numerator / denumerator

    sorted_bm25 = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
    return sorted_bm25


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)