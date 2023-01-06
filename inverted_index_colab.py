from collections import defaultdict, Counter
from contextlib import closing
import pickle
import operator
from operator import itemgetter
# from MultiFile import *
import hashlib
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import math
from pathlib import Path
import itertools

# stemmer
porterStemmer = PorterStemmer()
# download stopwords
nltk.download('stopwords')
BLOCK_SIZE = 1999998
NUM_BUCKETS = 124


# hash function
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in itertools.count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
    Parameters:
    -----------
      docs: dict mapping doc_id to list of tokens
    """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)
        # Document Number
        self._N = 0
        # Dict for document length
        self.doc2len = {}
        # dict for document VectorLength
        self.doc2vec_len = {}
        # dict to match doc_id to title for results
        self.doc2title = {}

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
        the tf of tokens, then update the index (in memory, no storage 
        side-effects).
    """
        self._N += 1
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file:
        (1) `name`.pkl containing the global term stats (e.g. df).
    """
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
        from the object's state dictionary. 
    """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
        a (word:str, [(doc_id:int, tf:int), ...]) tuple.
    """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    @staticmethod
    def write_a_posting_list(b_w_pl, folder):
        """ Takes a (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
        and writes it out to disk as files named {bucket_id}_XXX.bin under the
        current directory. Returns a posting locations dictionary that maps each
        word to the list of files and offsets that contain its posting list.
        Parameters:
        -----------
          b_w_pl: tuple
            Containing a bucket id and all (word, posting list) pairs in that bucket
            (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
        Return:
          posting_locs: dict
            Posting locations for each of the words written out in this bucket.
        """
        posting_locs = defaultdict(list)
        bucket, list_w_pl = b_w_pl

        with closing(MultiFileWriter(folder, bucket)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
        return posting_locs

    def read_posting_list(self, w):
        # stem = porterStemmer.stem(w)
        if w in self.df.keys() and self.posting_locs.keys():
            with closing(MultiFileReader()) as reader:
                locs = self.posting_locs[w]
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                return posting_list

    def get_idf(self, w):  # calculate the body tf. return list
        idf = math.log((self._N / self.df[w]), 2)
        return idf


def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


def partition_postings_and_write(postings, folder):
    """ A function that partitions the posting lists into buckets, writes out
    all posting lists in a bucket to disk, and returns the posting locations for
    each bucket. Partitioning should be done through the use of `token2bucket`
    above. Writing to disk should use the function  `write_a_posting_list`, a
    static method implemented in inverted_index_colab.py under the InvertedIndex
    class.
    Parameters:
    -----------
    postings: RDD
      An RDD where each item is a (w, posting_list) pair.
    Returns:
    --------
    RDD
      An RDD where each item is a posting locations dictionary for a bucket. The
      posting locations maintain a list for each word of file locations and
      offsets its posting list was written to. See `write_a_posting_list` for
      more details.
    """
    b_w_pl = postings.map(lambda x: (token2bucket_id(x[0]), (x[0], x[1]))).groupByKey()
    return b_w_pl.map(lambda x: InvertedIndex().write_a_posting_list(x, folder))


def word_count(text, id):
    """ Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    filtered_tokens = [tok for tok in tokens if (tok not in all_stopwords)]
    doc_word_count = Counter(filtered_tokens)
    return [(tok, (id, tf)) for tok, tf in doc_word_count.items()]


def get_doc_len(text, doc_id):
    """ Count document filtered length for storage in index as well as document vector length for RDD calculations
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (doc_id, doc_length, SumOfSquares(tf))
  """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    filtered_tokens = [tok for tok in tokens if (tok not in all_stopwords)]
    doc_word_count = Counter(filtered_tokens)
    doc_length = 0
    doc_vec_length = 0
    for tok, tf in doc_word_count.items():
        doc_length += tf
        doc_vec_length += tf ** 2
    return [(doc_id, doc_length, doc_vec_length)]


def reduce_word_counts(unsorted_pl):
    """ Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples
  Returns:
  --------
    list of tuples
      A sorted posting list.
  """
    sorted_pl = sorted(unsorted_pl, key=itemgetter(0))
    return sorted_pl


def calculate_df(postings):
    """ Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  """
    return postings.map(lambda x: (x[0], len(x[1])))
