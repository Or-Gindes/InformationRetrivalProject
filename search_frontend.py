from flask import Flask, request, jsonify
import pandas as pd
from collections import defaultdict, Counter
import re
import nltk
import pickle
import numpy as np
from math import sqrt
import os

nltk.download('stopwords')
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1
from inverted_index_colab import *

BODY_INDEX = 'train_body_index'
TITLE_INDEX = 'train_title_index'
ANCHOR_INDEX = 'train_anchor_index'
index_text = InvertedIndex().read_index(os.path.join(os.getcwd(), BODY_INDEX), BODY_INDEX)
index_title = InvertedIndex().read_index(os.path.join(os.getcwd(), TITLE_INDEX), TITLE_INDEX)
index_anchor = InvertedIndex().read_index(os.path.join(os.getcwd(), ANCHOR_INDEX), ANCHOR_INDEX)


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
ALL_STOPWORDS = stopwords_frozen.union(corpus_stopwords)


def tokenize(text):
    """
    This function turns text into a list of tokens. Moreover, it filters stopwords.

    Parameters:
        text: string , represting the text to tokenize.

    Returns:
         list of tokens (e.g., list of tokens).
    """

    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in ALL_STOPWORDS]
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
    query = 'apple' #request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # Tokenize query
    tokenized_query = tokenize(query)
    if len(tokenized_query) == 0:
        return jsonify(res)

    title_weight, body_weight = 0.5, 0.5
    merged_score = defaultdict(float)
    sorted_cosin_sim_body = cosin_similarity_score(tokenized_query, index_text)
    sorted_cosin_sim_title = cosin_similarity_score(tokenized_query, index_title)

    for doc_id, sim_score in sorted_cosin_sim_body.items():
        merged_score[doc_id] += sim_score * body_weight
    for doc_id, sim_score in sorted_cosin_sim_title.items():
        merged_score[doc_id] += sim_score * title_weight

    sorted_merged_score = {k: v for k, v in sorted(merged_score.items(), key=lambda item: item[1], reverse=True)}
    # add top 100 docs to result
    for i, doc_id in enumerate(sorted_merged_score.keys()):
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
          list of PageRank scores that correrspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # edges, vertices = generate_graph(pages_links)
    # edgesDF = edges.toDF(['src', 'dst']).repartition(4, 'src')
    # verticesDF = vertices.toDF(['id']).repartition(4, 'id')
    # g = GraphFrame(verticesDF, edgesDF)
    # pr_results = g.pageRank(resetProbability=0.15, maxIter=10)
    # pr = pr_results.vertices.select("id", "pagerank")
    # pr = pr.sort(col('pagerank').desc())
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
    # pages_ranked = [(title, index_anchor.pageViews[id]) for id, title, _ in pages]
    # pages_ranked.sort(key=lambda page: page[1], reverse=True)
    # END SOLUTION
    return jsonify(res)


def generate_graph(pages):
    """ Compute the directed graph generated by wiki links.
  Parameters:
  -----------
    pages: RDD
      An RDD where each row consists of one wikipedia articles with 'id' and
      'anchor_text'.
  Returns:
  --------
    edges: RDD
      An RDD where each row represents an edge in the directed graph created by
      the wikipedia links. The first entry should the source page id and the
      second entry is the destination page id. No duplicates should be present.
    vertices: RDD
      An RDD where each row represents a vetrix (node) in the directed graph
      created by the wikipedia links. No duplicates should be present.
  """
    edges = pages.map(lambda page: [(page[0], link_id.id) for link_id in page[1]]).flatMap(lambda ls: ls).distinct()
    vertices = edges.map(lambda edge: [edge[0], edge[1]]).flatMap(lambda ls: ls).distinct().map(lambda x: (x,))
    return edges, vertices


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
    query_vec_len = sum([c ** 2 for w, c in tf_query.items()])
    for term, count in tf_query.items():
        pls = index.read_posting_list(term, index_folder)
        term_idf = index.get_idf(term)
        for doc_id, doc_tf in pls:
            # normalized query tfidf
            query_tfidf = count / query_len * term_idf
            # normalized document tfidf
            doc_tfidf = doc_tf / index.doc2len[doc_id] * term_idf
            cosine_sim_numerator[doc_id] += doc_tfidf * query_tfidf

    # vector length of each doc is calculated at index creation
    cosine_sim = {doc_id: numerator / sqrt(index.doc2vec_len[doc_id] * query_vec_len) for doc_id, numerator in
                  cosine_sim_numerator.items()}
    sorted_cosin_sim = {k: v for k, v in sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)}
    return sorted_cosin_sim


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
