"""Module providing a utils functions for the project."""


from collections import Counter
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer


def count_categories(category_tree: str) -> int:
    """Counts number of sub categories within a category tree

    Parameters
    ----------
    category_tree : str
        string of the category tree to anayse

    Returns
    -------
    _type_
        Number of sub categories
    """
    category_tree = category_tree.replace('["', '')
    category_tree = category_tree.replace('"]', '')
    category_tree = category_tree.replace('>>', '>')
    category_tree = category_tree.split(" > ")

    for category in category_tree:
        if "..." in category:
            category_tree.remove(category)
    return len(category_tree)


def find_dots(string: str) -> bool:
    """Indicates if a string countains "..."

    Parameters
    ----------
    string : str
        String to check

    Returns
    -------
    bool
    """
    if "..." in string:
        return True
    return False


def extract_label(category_tree: str, position: int) -> str:
    """Returns the category corresponding to the position.
    For instance if category_tree="Home >> Kitchen" and position=2
    the function returns Kitchen.
    If "..." in the category, this one is not returned

    Parameters
    ----------
    category_tree : str
        Category tree of categories
    position : int
        position of the category we want to extract

    Returns
    -------
    str
        _description_
    """
    category_tree = category_tree.replace('["', '')
    category_tree = category_tree.replace('"]', '')
    category_tree = category_tree.replace('>>', '>')
    category_tree = category_tree.split(" > ")

    if len(category_tree) < position + 1:
        return "/"
    category = category_tree[position]
    if "..." in category:
        return "/"
    return category


def tokenize(doc: str) -> list:
    """Tokize a text into a list of words

    Parameters
    ----------
    doc : str
        Text to tokenize

    Returns
    -------
    list
        list of tokens
    """
    # Init
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Tokenize the document
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(doc)
    # Lemmatize and remove stop words
    words = [
        lemmatizer.lemmatize(word.lower(), pos="v")
        for word in words
        if word.lower() not in stop_words
    ]
    return words


def word_vectorizer(corpus: list, tfidf: bool = False) -> np.array:
    """Vectorizes a corpus

    Parameters
    ----------
    corpus : list
        _description_
    tfidf : bool, optional
        _description_, by default False

    Returns
    -------
    np.array
        _description_
    """
    # Creates unique list of words sorted
    bow = list(set([word for text in corpus for word in text]))
    bow.sort()
    # Fill row by row array
    res = np.zeros((len(corpus), len(bow)), dtype=np.uint8)
    for i in range(len(corpus)):
        words_count = [word for word in corpus[i]]
        words_count = Counter(words_count)
        words_count = dict(
            sorted(words_count.items(),
                   key=lambda item: item[1],
                   reverse=True))
        for j, word in enumerate(bow):
            if word in words_count.keys():
                res[i][j] = words_count[word]
            else:
                res[i][j] = 0
    if tfidf:
        transformer = TfidfTransformer()
        res = transformer.fit_transform(res).toarray()
    return res, np.array(bow)
