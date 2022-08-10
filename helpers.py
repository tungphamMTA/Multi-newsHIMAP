# story_separator_special_tag

from nltk.tokenize import sent_tokenize, word_tokenize
import unicodedata
from nltk.tokenize.treebank import TreebankWordDetokenizer
# https://stackoverflow.com/questions/56856394/rejoin-sentence-like-original-after-tokenizing-with-nltk-word-tokenize
from nltk.tokenize import word_tokenize
from nltk import WordPunctTokenizer



def split_doc(text, max_length):
    """
    doc: document >=4096 word
    max_length: length maximun to input summary
    output: split doc to mul small doc
    """

    word_tokens_split = word_tokenize(text)
    if len(word_tokens_split)<=max_length:
        return [text]
    sentence_tokens = sent_tokenize(text)

    length_text = len(word_tokens_split)
    doc_split_num = length_text // max_length + 1

    word_in_sub_doc = length_text // doc_split_num
    count = 0
    text  = ""
    result = []
    count_doc = 0
    for sentence in sentence_tokens:
        count += len(word_tokenize(sentence))
        if count > word_in_sub_doc:
            count = 0
            count_doc +=1
            result.append(text)
            text = ""
        text += " ".join(sentence.split()) +" "
    # append last doc
    if count_doc < doc_split_num:
        result.append(text)
    return result

def preprocess(s):
    s= s.lower()
    wpt = WordPunctTokenizer()
    w_tokens = wpt.tokenize(s)
    # w_tokens = word_tokenize(s)
    return " ".join(text for text in w_tokens)

def concate_text(list_raw_text):
    concate_txt = ""
    for raw_text in list_raw_text:
        concate_txt += (raw_text + " story_separator_special_tag ")
    return concate_txt

if __name__ == '__main__':
    text ="I'm here story_separator_special_tag what time is it ???"
    print(preprocess(text))
    