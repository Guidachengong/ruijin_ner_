# encoding=utf8

import codecs
import os

import numpy as np

filepath = "./datasets/backup"
file = "./data/example.train"


def change_format(filepath):
    sentences = []
    sentence = []
    files = os.listdir(filepath)
    for f in files:
        taglist = []
        if f.endswith("ann"):
            f_name = f.split(".")[0]
            for char in codecs.open(os.path.join(filepath, f), "r", "utf-8"):
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

            c2_index = np.argsort(taglist_c2)
            taglist_new = [taglist[i] for i in c2_index]
            taglist_c2.sort()
            f_txt = codecs.open(os.path.join(filepath, f_name + ".txt"), "r", "utf-8")

            text = f_txt.read()
            doc = [[] for _ in range(len(text))]
            for info in taglist_new:
                doc[info[1]] = [text[info[1]], info[0]]
            for i in range(len(doc)):
                if not doc[i]:
                    doc[i] = [text[i], 'O']
                if doc[i][0] == "。":
                    sentence.append(doc[i])
                    label = [cha[-1] for cha in sentence]
                    if label.count('O') != len(sentence):
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


def read_ann(file):
    i = 1
    for f in codecs.open(file, "r", "utf-8"):
        # print(f)
        if i == 41:
            print(f)
            print(f.split("\t")[2].find(" "))
        i += 1


if __name__ == '__main__':
    sent = change_format(filepath)
    for ch in sent:
        print(ch)
    # print(np.array(sent))
    # print(len(sent[1]))
    # read_ann(filepath)
