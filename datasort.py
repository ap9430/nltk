__author__ = 'parkerat2'

import pandas as pd
import csv
import shutil
import nltk as nl
import nltk.corpus as nc
import re as regex
import math
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np


#def datasplit(size, train, cv, test):
#    trainSize = size * train

#def nltk(messcol):

def linkclean(csvpath):
    features = []
    csv = pd.read_csv(csvpath)

    mes = csv['message']
    for i in range(len(mes)):
        if "http" in mes[i] or "www" in mes[i]:
            mes[i] = regex.sub("http\S+", "", mes[i])
            mes[i] = regex.sub("www\S+", "", mes[i])
            features.append(1)
        else:
            features.append(0)

    csv['message'] = mes
    csv.to_csv("..\data\linksremoved.csv", index=False)

    return features


#http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/
def languagedetect(text):
    lang_ratios = {}
    text = text.lower()
    tokens = nl.wordpunct_tokenize(text)

    for lang in nc.stopwords.fileids():
        stopwords_set = set(nc.stopwords.words(lang))
        words_set = set(tokens)
        common_elements = words_set.intersection(stopwords_set)

        lang_ratios[lang] = len(common_elements)
    containsSpanish = lang_ratios.__contains__("'spanish:' 0")

    return containsSpanish


def languageclean(csvpath):
    df = []
    #csv = pd.read_csv(csvpath)
    #text = csv['message']
    fieldnames = ['Coding:Level1', 'Coding:Level2', 'message']
    with open(csvpath, 'r') as csvfile, open('..\data\languageclean.csv', 'w', newline='') as outputfile:
        reader = csv.DictReader(csvfile, fieldnames = fieldnames)
        writer = csv.DictWriter(outputfile, fieldnames = fieldnames)
        for row in reader:
            if languagedetect(row['message']) == False:
                writer.writerow({'Coding:Level1': row['Coding:Level1'],'Coding:Level2': row['Coding:Level2'],'message': row['message']})

    #df.to_csv('../data/languageclean.csv')


def featureExtractor(text):
    tokens = nl.word_tokenize(text)
    pos = nl.pos_tag(tokens)

    #http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    tagset = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',	'MD', 'NN',	'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
              'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',	'WDT', 'WP', 'WP$', 'WRB']

    features = []
    length = len(pos)

    for tag in tagset:
        count = 0
        for i in pos:
            if tag in i:
                count += 1
        features.append(count)

    features2 = []

    for i in features:
        if length != 0:
            freq = i / length
            features2.append(freq)
        else:
            features2.append(0)

    for i in features2:
        features.append(i)

    features.append(length)

    keywords = ['donation', 'donations', 'donate', 'bid', 'auction', 'prize', 'weekend', 'tomorrow', 'morning', 'today', 'tonight',
                'events', 'else', 'petition', 'lobby', 'lobbying', 'lobbyist', 'sign', 'help', 'looking', 'members', 'member',
                'store', 'sell', 'sold', 'available', 'buy', 'shop', 'shopping', 'new', 'product', 'selling', 'time', 'volunteer',
                'volunteers', 'thanks', 'you', 'thank', 'good', 'great', 'job', 'work']

    text = text.lower()
    for k in keywords:
        if text.__contains__(k):
            features.append(1)
        else:
            features.append(0)

    return features


def logReg(pickle, dataframe, randarray, iterations):

    messages = dataframe['message']
    l1 = dataframe['Coding:Level1']
    l2 = dataframe['Coding:Level2']

    X = pickle
    y = l1
    y2 = l2
    sc = StandardScaler()
    lr = LogisticRegression(C = 1000.0, random_state=0)
    recallf1 = []
    recallf2 = []

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)
    # lr.fit(X_train_std, y_train)
    # y_pred = lr.predict(X_test_std)
    #
    # X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.01, random_state=0)
    # sc.fit(X2_train)
    # X2_train_std = sc.transform(X2_train)
    # X2_test_std = sc.transform(X2_test)
    # lr.fit(X2_train_std, y2_train)
    # y2_pred = lr.predict(X2_test_std)
    #
    # print(recall_score(y_test, y_pred, average='micro'))
    # print(recall_score(y2_test, y2_pred, average='micro'))

    for split in randarray:

        prob1 = 0
        prob2 = 0

        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            lr.fit(X_train_std, y_train)
            y_pred = lr.predict(X_test_std)

            prob1 += recall_score(y_test, y_pred, average='micro')

            X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=split)
            sc.fit(X2_train)
            X2_train_std = sc.transform(X2_train)
            X2_test_std = sc.transform(X2_test)
            lr.fit(X2_train_std, y2_train)
            y2_pred = lr.predict(X2_test_std)

            prob2 += recall_score(y2_test, y2_pred, average='micro')

        recall1 = prob1/iterations * 100
        recallf1.append(recall1)
        recall2 = prob2/iterations * 100
        recallf2.append(recall2)

        #print("Category 1 Logistic Regression Test size: %.2f Recall: %.2f" % (split, recall1))
        #print("Category 2 Logistic Regression Test size: %.2f Recall: %.2f" % (split, recall2))

    return recallf1, recallf2


def svm(pickle, dataframe, randarray, iterations, kern):
    messages = dataframe['message']
    l1 = dataframe['Coding:Level1']
    l2 = dataframe['Coding:Level2']

    gam = 0.0
    if kern is 'rbf':
        gam = 'auto'

    X = pickle
    y = l1
    y2 = l2
    sc = StandardScaler()

    sv = SVC(kernel=kern, C = 1.0, gamma=gam, random_state=0)
    recallf1 = []
    recallf2 = []

    for split in randarray:

        prob1 = 0
        prob2 = 0

        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            sv.fit(X_train_std, y_train)
            y_pred = sv.predict(X_test_std)

            prob1 += recall_score(y_test, y_pred, average='micro')

            X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=split)
            sc.fit(X2_train)
            X2_train_std = sc.transform(X2_train)
            X2_test_std = sc.transform(X2_test)
            sv.fit(X2_train_std, y2_train)
            y2_pred = sv.predict(X2_test_std)

            prob2 += recall_score(y2_test, y2_pred, average='micro')

        recall1 = prob1/iterations * 100
        recallf1.append(recall1)
        recall2 = prob2/iterations * 100
        recallf2.append(recall2)

        #print("Category 1 Support Vector Machine Test size: %.2f Recall: %.2f" % (split, recall1))
        #print("Category 2 Support Vector Machine Test size: %.2f Recall: %.2f" % (split, recall2))

    return recallf1, recallf2


def decTree(pickle, dataframe, randarray, iterations, crit):
    messages = dataframe['message']
    l1 = dataframe['Coding:Level1']
    l2 = dataframe['Coding:Level2']

    X = pickle
    y = l1
    y2 = l2
    sc = StandardScaler()

    tree = DecisionTreeClassifier(criterion=crit, max_depth=None, random_state=0)
    recallf1 = []
    recallf2 = []

    for split in randarray:

        prob1 = 0
        prob2 = 0

        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            tree.fit(X_train_std, y_train)
            y_pred = tree.predict(X_test_std)

            prob1 += recall_score(y_test, y_pred, average='micro')

            X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=split)
            sc.fit(X2_train)
            X2_train_std = sc.transform(X2_train)
            X2_test_std = sc.transform(X2_test)
            tree.fit(X2_train_std, y2_train)
            y2_pred = tree.predict(X2_test_std)

            prob2 += recall_score(y2_test, y2_pred, average='micro')

        recall1 = prob1/iterations * 100
        recallf1.append(recall1)
        recall2 = prob2/iterations * 100
        recallf2.append(recall2)

        #print("Category 1 Decision Tree Test size: %.2f Recall: %.2f" % (split, recall1))
        #print("Category 2 Decision Tree Test size: %.2f Recall: %.2f" % (split, recall2))

    return recallf1, recallf2


def knn(pickle, dataframe, randarray, iterations, num):
    messages = dataframe['message']
    l1 = dataframe['Coding:Level1']
    l2 = dataframe['Coding:Level2']

    X = pickle
    y = l1
    y2 = l2
    sc = StandardScaler()

    kn = KNeighborsClassifier(n_neighbors=num)
    recallf1 = []
    recallf2 = []

    for split in randarray:

        prob1 = 0
        prob2 = 0

        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            kn.fit(X_train_std, y_train)
            y_pred = kn.predict(X_test_std)

            prob1 += recall_score(y_test, y_pred, average='micro')

            X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=split)
            sc.fit(X2_train)
            X2_train_std = sc.transform(X2_train)
            X2_test_std = sc.transform(X2_test)
            kn.fit(X2_train_std, y2_train)
            y2_pred = kn.predict(X2_test_std)

            prob2 += recall_score(y2_test, y2_pred, average='micro')

        recall1 = prob1/iterations * 100
        recallf1.append(recall1)
        recall2 = prob2/iterations * 100
        recallf2.append(recall2)

        #print("Category 1 K Nearest Neighbors Test size: %.2f Recall: %.2f" % (split, recall1))
        #print("Category 2 K Nearest Neighbors Test size: %.2f Recall: %.2f" % (split, recall2))

    return recallf1, recallf2


def naiveBayes(pickle, dataframe, randarray, iterations):
    messages = dataframe['message']
    l1 = dataframe['Coding:Level1']
    l2 = dataframe['Coding:Level2']

    X = pickle
    y = l1
    y2 = l2
    #sc = StandardScaler()

    mnb = MultinomialNB()
    recallf1 = []
    recallf2 = []

    for split in randarray:

        prob1 = 0
        prob2 = 0

        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
            #sc.fit(X_train)
            #X_train_std = sc.transform(X_train)
            #X_test_std = sc.transform(X_test)
            mnb.fit(X_train, y_train)
            y_pred = mnb.predict(X_test)

            prob1 += recall_score(y_test, y_pred, average='micro')

            X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=split)
            #sc.fit(X2_train)
            #X2_train_std = sc.transform(X2_train)
            #X2_test_std = sc.transform(X2_test)
            mnb.fit(X2_train, y2_train)
            y2_pred = mnb.predict(X2_test)

            prob2 += recall_score(y2_test, y2_pred, average='micro')

        recall1 = prob1/iterations * 100
        recallf1.append(recall1)
        recall2 = prob2/iterations * 100
        recallf2.append(recall2)

        #print("Category 1 Naive Bayes Test size: %.2f Recall: %.2f" % (split, recall1))
        #print("Category 2 Naive Bayes Test size: %.2f Recall: %.2f" % (split, recall2))

    return recallf1, recallf2


def classifierComparer(classifiers, feat, df, testsplit):
    y_label = "Recall"
    x_label = "Percent of Test"
    iterations = 10
    plt.clf()
    colors = ['blue', 'red']

    count = 1
    labels = []
    for j in classifiers:
        for i in range(1, 3):
            labels.append(str(j) + '-' + str(i))

    for f in feat:
        for clss in classifiers:
            recall1, recall2 = naiveBayes(f, df, testsplit, iterations)
            # if clss is classifiers[0]:
            #     plt.plot(testsplit, recall1, color='red', label=str(str(clss) + "-" + str(1)))
            #     plt.plot(testsplit, recall2, color='red', linestyle= "--", label=str(str(clss) + "-" + str(2)))
            # elif clss is classifiers[1]:
            #     plt.plot(testsplit, recall1, color='blue', label=str(str(clss) + "-" + str(1)))
            #     plt.plot(testsplit, recall2, color='blue', linestyle= "--", label=str(str(clss) + "-" + str(2)))
            # elif clss is classifiers[2]:
            #     plt.plot(testsplit, recall1, color='green', label=str(str(clss) + "-" + str(1)))
            #     plt.plot(testsplit, recall2, color='green', linestyle= "--", label=str(str(clss) + "-" + str(2)))
            # elif clss is classifiers[3]:
            #     plt.plot(testsplit, recall1, color='black', label=str(str(clss) + "-" + str(1)))
            #     plt.plot(testsplit, recall2, color='black', linestyle= "--", label=str(str(clss) + "-" + str(2)))
            # elif clss is classifiers[4]:
            #     plt.plot(testsplit, recall1, color='magenta', label=str(str(clss) + "-" + str(1)))
            #     plt.plot(testsplit, recall2, color='magenta', linestyle= "--", label=str(str(clss) + "-" + str(2)))
            # elif clss is classifiers[5]:
            #     plt.plot(testsplit, recall1, color='yellow', label=str(str(clss) + "-" + str(1)))
            #     plt.plot(testsplit, recall2, color='yellow', linestyle= "--", label=str(str(clss) + "-" + str(2)))
            # elif clss is classifiers[6]:
            #     plt.plot(testsplit, recall1, color='cyan', label=str(str(clss) + "-" + str(1)))
            #     plt.plot(testsplit, recall2, color='cyan', linestyle= "--", label=str(str(clss) + "-" + str(2)))

            plt.plot(testsplit, recall1, color='blue', label=str(str(clss) + str(1)))
            plt.plot(testsplit, recall2, color='red', label=str(str(clss) + str(2)))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(labels, loc="best")
        plt.title("FeatureSet "+ str(count))
        plt.savefig("../files/NB_FeatureSet" + str(count) + ".png")
        plt.clf()
        count += 1


def main():
    #linkfeats = []
    #
    #linkfeats = linkclean('E:\PycharmProjects\sem\data\labelsmessages.csv')
    # #languageclean('E:\PycharmProjects\sem\data\linksremoved.csv')
    #
    # features = []
    # count = 0
    # for mes in messages:
    #     if not isinstance(mes, str):
    #         if math.isnan(mes):
    #             mes = ""
    #
    #     features.append(featureExtractor(mes))
    #     #count += 1
    #
    # for feat in features:
    #     x = linkfeats.pop(0)
    #     feat.append(x)
    #
    # print(features[0])
    # print(len(features[0]))

    df = pd.read_csv('E:\PycharmProjects\sem\data\linksremoved.csv')

    # pd.to_pickle(features, '..\data\listfeats.pkl')
    featureset1 = pd.read_pickle('E:\PycharmProjects\sem\data\listfeats.pkl')

    #pd.to_pickle(features, '..\data\listfeats2.pkl')
    featureset2 = pd.read_pickle('E:\PycharmProjects\sem\data\listfeats2.pkl')

    testsplit = [0.30, 0.20, 0.10, 0.05, 0.01]
    iterations = 10
    kernel1 = "rbf"
    kernel2 = "linear"
    crit1 = "gini"
    crit2 = "entropy"
    numneighbors = 12

    #logrecall1, logrecall2 = logReg(featureset1, df, testsplit, iterations)
    #svmrecall1, svmrecall2 = svm(featureset1, df, testsplit, iterations, kernl)
    #decrecall1, decrecall2 = decTree(featureset1, df, testsplit, iterations, crit)
    #knnrecall1, knnrecall2 = knn(featureset2, df, testsplit, iterations, numneighbors)
    #gnbrecall1, gnbrecall2 = naiveBayes(featureset1, df, testsplit, iterations)

    feat = [featureset1, featureset2]
    numneighbors = [5, 7, 9, 10, 11, 13, 15]
    classifiers = ["Log Reg", "SVM", "Dec Tree", "KNN"]
    kernel = [kernel1, kernel2]
    criterion = [crit1, crit2]
    log = ['Level']
    nb = ['Level']

    classifierComparer(nb, feat, df, testsplit)


if __name__ == '__main__':
    main()
