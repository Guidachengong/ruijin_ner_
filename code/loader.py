import os
import re
import codecs

import numpy as np
from itertools import product

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    files = os.listdir(path)
    for f in files:
        taglist = []
        if f.endswith("ann"):
            f_name = f.split(".")[0]
            for char in codecs.open(os.path.join(path, f), "r", "utf-8"):
                char_list_old = char.split("\t")[1].split(" ")
                char_list = [char_list_old[0], char_list_old[1], char_list_old[-1]]
                taglist.append(char_list)
            try:
                taglist = [["B-" + a[0], int(a[1]), int(a[2])] for a in taglist]
            except:
                print(f)
            taglist_all = taglist
            for ele in taglist:
                temp = ele[1] + 1
                while temp < ele[2]:
                    word = ["I-" + ele[0].replace("B-", ""), temp, 0]
                    temp += 1
                    taglist_all.append(word)

            taglist_c2 = np.array([x[1] for x in taglist])

            # print(taglist_c2)

            c2_index = np.argsort(taglist_c2)

            # print(c2_index)
            taglist_new = [taglist[i] for i in c2_index]
            taglist_c2.sort()
            f_txt = codecs.open(os.path.join(path, f_name + ".txt"), "r", "utf-8")

            text = f_txt.read()
            doc = [[] for _ in range(len(text))]
            for info in taglist_new:
                doc[info[1]] = [text[info[1]], info[0]]
            for i in range(len(doc)):
                if not doc[i]:
                    doc[i] = [text[i], 'O']
                if doc[i][0] == "。":
                    sentence.append(doc[i])
                    # label = [cha[-1] for cha in sentence]
                    # if label.count('O') != len(sentence):
                    sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(doc[i])

    new = []
    for s in sentences:
        l = 0
        while l < len(s):
            if s[l][0] == '\n':
                del s[l]
            else:
                l += 1
        new.append(s)

    return new


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)


if __name__ == '__main__':


    path = '../data/log'
    arr = load_sentences(path)
    # ll = [len(x) for x in arr]
    # print(max(ll))
    # print(arr[0])
