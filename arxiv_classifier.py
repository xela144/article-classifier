#!/usr/bin/env python
# coding: utf-8
from pdb import set_trace as pp

# classifier, numerical imports
import gensim
import gensim.models.doc2vec
from gensim.models import Doc2Vec#, tfidf
import numpy as np
import statsmodels.api as sm

# other imports
from contextlib import contextmanager
from timeit import default_timer
import multiprocessing
from collections import namedtuple
import locale
from random import sample, shuffle
from collections import defaultdict
import datetime
import random

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

# database connector
from db import psqlServer

# Names of databases for SQL select
table_names = dict(
    AI = "arx_AI"       , # artificial intelligence
    FA = "arx_math_FA"  , # functional analysis
    GR = "arx_GRQC"     , # general relativity and quantum cosmology
    LG = "arx"          , # learning
    NT = "arx_math_NT"  , # number theory
)
# select-type statements
UNION = " UNION ALL "
SEL_C_FOO = "SELECT COUNT(*) as FOO FROM "
SEL_C = "SELECT COUNT(*) FROM "
SEL_STAR = "SELECT * FROM "

control_chars = [chr(0x85)]
locale.setlocale(locale.LC_ALL, 'C')

def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    norm_text = norm_text.replace('_', '-')
    norm_text = norm_text.replace('\n', ' ')

    # ok just for math articles this list will eventually cover all the greek letters -- no thanks:
    # norm_text = norm_text.replace(u'\xe2', 'a')
    # norm_text = norm_text.replace(u'\xe8', 'e')
    # norm_text = norm_text.replace(u'\xe9', 'e')
    # norm_text = norm_text.replace(u'\xea', 'e')
    # norm_text = norm_text.replace(u'\xf6', 'o')
    # norm_text = norm_text.replace(u'\xfc', 'u')
    # norm_text = norm_text.replace(u'\u03b1', 'alpha')
    # norm_text = norm_text.replace(u'\u03b2', 'beta')
    # norm_text = norm_text.replace(u'\u03b3', 'gamma')
    # norm_text = norm_text.replace(u'\u03b5', 'epsilon')
    # norm_text = norm_text.replace(u'\u03b6', 'cau')
    # norm_text = norm_text.replace(u'\u03c0', 'pi')
    # norm_text = norm_text.replace(u'\u03c6', 'phi')
    # norm_text = norm_text.replace(u'\u03c9', 'omega')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text

# How many items in all tables?
def get_table_counts(conn):
    sum_statement = "SELECT SUM(FOO) FROM ("
    sum_statement += SEL_C_FOO + table_names['AI']
    sum_statement += UNION + SEL_C_FOO + table_names['FA']
    sum_statement += UNION + SEL_C_FOO + table_names['GR']
    sum_statement += UNION + SEL_C_FOO + table_names['LG']
    sum_statement += UNION + SEL_C_FOO + table_names['NT']
    sum_statement += ") as B;"
    total = conn.execute(sum_statement)[0]['sum']
    print(total)

    # Per table
    total_AI = conn.execute(SEL_C + table_names['AI'])[0]['count']
    total_FA = conn.execute(SEL_C + table_names['FA'])[0]['count']
    total_GR = conn.execute(SEL_C + table_names['GR'])[0]['count']
    total_LG = conn.execute(SEL_C + table_names['LG'])[0]['count']
    total_NT = conn.execute(SEL_C + table_names['NT'])[0]['count']

    print("AI:{}".format(total_AI))
    print("FA:{}".format(total_FA))
    print("GR:{}".format(total_GR))
    print("LG:{}".format(total_LG))
    print("NT:{}".format(total_NT))


def get_corpus(conn, table, lim):
    statement = SEL_STAR + table + " ORDER BY id" + " LIMIT " + str(lim)
    # Get some rows from the AI table
    rows = conn.execute(statement, table, lim)
    corpus = [x['abstract'] for x in rows]
    labels = [float(x['has_journal_ref']) for x in rows]
    titles = [x['title'] for x in rows]
    # num_pages = [x['num_pages'] for x in rows]
    # num_figs = [x['num_figs'] for x in rows]
    authors = [x['author'] for x in rows]
    summary_length = [int(x['summary_length']) for x in rows]
    word_count = [int(x['summary_wc']) for x in rows]
    id_ = [int(x['id']) for x in rows]

    corpus_normalized = []
    jj = 0
    for corp in corpus:
        corpus_normalized.append(u"_*{0} {1}\n".format(jj, normalize_text(corp)))
        jj += 1
    titles_normalized = []
    jj = 0
    for title in titles:
        titles_normalized.append(u"{}".format(normalize_text(title)))
        jj += 1


    return corpus_normalized, labels, titles_normalized, authors, summary_length, word_count, id_


def get_all_docs(corpus, labels, titles):
    """ Break the corpus in to following parts:

    First fourth is training set.
    Second fourth is test set
    Second half is 'extra'

    """
    ll = len(corpus)
    quarter_len = ll//4
    half_len = ll//2

    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment title')
    alldocs = []


    for line_no, line in enumerate(corpus[0:quarter_len]):
        tokens = gensim.utils.to_unicode(corpus[line_no]).split()
        words = tokens[1:]
        tags = [line_no]
        split = 'train'
        #split = ['train', 'test', 'extra', 'extra'][line_no//12500]
        sentiment = labels[line_no]
        title = titles[line_no]
        #sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//6750]
        alldocs.append(SentimentDocument(words, tags, split, sentiment, title))

    for line_no, line in enumerate(corpus[quarter_len:half_len]):
        tokens = gensim.utils.to_unicode(corpus[line_no]).split()
        words = tokens[1:]
        tags = [line_no]
        split = 'test'
        #split = ['train', 'test', 'extra', 'extra'][line_no//12500]
        sentiment = labels[line_no]
        title = titles[line_no]
        #sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//6750]
        alldocs.append(SentimentDocument(words, tags, split, sentiment, title))
        # alldocs.append(SentimentDocument(words, tags, split, sentiment))

    for line_no, line in enumerate(corpus[half_len:]):
        tokens = gensim.utils.to_unicode(corpus[line_no]).split()
        words = tokens[1:]
        tags = [line_no]
        split = 'extra'
        #split = ['train', 'test', 'extra', 'extra'][line_no//12500]
        sentiment = None
        title = titles[line_no]
        #sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//6750]
        alldocs.append(SentimentDocument(words, tags, split, sentiment, title))
        # alldocs.append(SentimentDocument(words, tags, split, sentiment))

    return alldocs

def get_models(alldocs):
    negative = 5
    min_count = 2
    size = 100
    hier_softmax = 0
    simple_models = [
        # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=size, window=5,  negative=negative, hs=hier_softmax, min_count=min_count, workers=cores),
        # PV-DBOW
        Doc2Vec(dm=0,              size=size,            negative=negative, hs=hier_softmax, min_count=min_count, workers=cores),
        # PV-DM w/ average
        Doc2Vec(dm=1, dm_mean=1,   size=size, window=10, negative=negative, hs=hier_softmax, min_count=min_count, workers=cores),
    ]
    '''         `iter` = number of iterations (epochs) over the corpus. The default inherited from Word2Vec is 5,
    but values of 10 or 20 are common in published 'Paragraph Vector' experiments.
    '''

    simple_models[0].build_vocab(alldocs)
    # print(simple_models[0])
    for model in simple_models[1:]:
        """Reuse shareable structures from other_model."""
        model.reset_from(simple_models[0])
        # print(model)

    #models_by_name = OrderedDict((str(model), model) for model in simple_models)
    #models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    #models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    # return simple_models, models_by_name
    # return models_by_name
    return simple_models


# Let's define some helper methods for evaluating the performance of our Doc2vec
# using paragraph vectors. We will classify document sentiments using a logistic
# regression model based on our paragraph embeddings. We will compare the error
# rates based on word embeddings from our various Doc2vec models.
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)#disp=0)
    # print(predictor.summary())
    return predictor

def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)

    # Here is the wtf: test_data is just [None]*500!!!
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)


def run_models(simple_models, doc_list, epochs=3):
# def run_models(models_by_name, epochs=3):

    # Predictive Evaluation Methods

    # Bulk Training

    # We use an explicit multiple-pass, alpha-reduction approach as sketched in
    # this gensim doc2vec blog post with added shuffling of corpus on each pass.
    # https://rare-technologies.com/doc2vec-tutorial/
    best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved

    alpha, min_alpha, passes = (0.025, 0.001, epochs)
    alpha_delta = (alpha - min_alpha) / passes

    print("START %s" % datetime.datetime.now())

    for epoch in range(passes):
        shuffle(doc_list)  # Shuffling gets best results

        for train_model in simple_models:
            # Train
            duration = 'na'
            train_model.alpha, train_model.min_alpha = alpha, alpha
            with elapsed_timer() as elapsed:
                train_model.train(doc_list, total_examples=len(doc_list), epochs=1)
                duration = '%.1f' % elapsed()

            # Evaluate
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs)
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if err <= best_error[str(train_model)]:
                best_error[str(train_model)] = err
                best_indicator = '*'
            print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, str(train_model), duration, eval_duration))

            if ((epoch + 1) % 5) == 0 or epoch == 0:
                eval_duration = ''
                with elapsed_timer() as eval_elapsed:
                    infer_err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs, infer=True)
                eval_duration = '%.1f' % eval_elapsed()
                best_indicator = ' '
                if infer_err < best_error[str(train_model) + '_inferred']:
                    best_error[str(train_model) + '_inferred'] = infer_err
                    best_indicator = '*'
                print("%s%f : %i passes : %s %ss %ss" % (best_indicator, infer_err, epoch + 1, str(train_model) + '_inferred', duration, eval_duration))

        print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

    print("END %s" % str(datetime.datetime.now()))

    return best_error


def results(best_error, simple_models, alldocs, table):
    # Print best error rates achieved
    print("Err rate Model")
    f = open('save_models/' + args.table + '/results.txt', encoding='utf-8', mode='a')
    f.write('Err rate model\n')
    for rate, name in sorted((rate, name) for name, rate in best_error.items()):
        print("%f %s" % (rate, name))
        f.write("%f %s\n" % (rate, name))

    doc_id = np.random.randint(simple_models[0].docvecs.count)  # Pick random doc; re-run cell for more examples
    out = 'for doc_id %d %s <db_id:%d> ' % (doc_id, alldocs[doc_id].title, alldocs[doc_id].tags[0])
    print(out)
    f.write(out)

    for model in simple_models:
        inferred_docvec = model.infer_vector(alldocs[doc_id].words)
        top3 = model.docvecs.most_similar([inferred_docvec], topn=3)

        out = '\n\nMODEL: %s' % model
        print(out)
        f.write(out)
        for top in top3:
            out ='\n %s, %f, %s' % (top[0], top[1], alldocs[top[0]].title)
            print(out)
            f.write(out)

    f.write("\n\n\n")

    doc_id = np.random.randint(simple_models[0].docvecs.count)  # pick random doc, re-run cell for more examples
    model = random.choice(simple_models)  # and a random model
    sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
    out = u'TARGET (%d): «%s»\n' % (doc_id, ' '.join(alldocs[doc_id].words))
    print(out)
    f.write(out)
    f.write("\n")
    out = u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model
    print(out)
    f.write(out)
    f.write("\n")
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        out = u'%s %s: «%s»\n' % (label, sims[index], ' '.join(alldocs[sims[index][0]].words))
        print(out)
        f.write(out + '\n\n')
    f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load',  nargs=1, dest='load', action='store_const',
    #                     help='load Doc2Vec format corpus')
    parser.add_argument('-l', '--load', action='store_true', help='load Doc2Vec format corpus')
    parser.add_argument('-t', '--table', default='LG', help='choose table name')
    parser.add_argument('-n', '--num', default=10000, help='number of articles to load')
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-c', '--save-corpus', action='store_true')
    parser.add_argument('-d', '--save-dict', action='store_true')
    parser.add_argument('-p', '--do-plots', action='store_true')


    args = parser.parse_args()

    conn = psqlServer()
    # get_table_counts(conn)

    table_name = table_names[args.table]
    num_articles = int(args.num)

    f = open('save_models/' + args.table + '/results.txt', encoding='utf-8', mode='w')
    out = "getting corpus from table={} number of articles={}".format(table_name, num_articles)
    print(out)
    f.write(out)

    # dump info from db
    corpus, labels, titles, authors, summary_length, word_count, id_ = get_corpus(conn, table_name, num_articles)

    publish_rate = sum(labels)/num_articles
    f.write("\n publish_rate:{}\n".format(publish_rate))
    f.close()

    alldocs = get_all_docs(corpus, labels, titles)
    texts = [x.words for x in alldocs]

    if args.save_corpus:
        with open('save_models/' + args.table + '/corpus.txt', encoding='utf-8', mode='w') as corp:
            i = 0
            # corp.write("_{}_".format(publish_rate) + "_{}_{}_{}*\n".format(args.table, num_articles, int(sum(labels))))
            for line in texts:
                corp.write(u"_{}_{}_{}*".format(int(labels[i]), titles[i], id_[i]) + u" ".join(line) + '\n')
                i += 1

    allwords = [word for sublist in texts for word in sublist]
    print("there were {} words in this dataset".format(len(allwords)))
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    dictionary = gensim.corpora.Dictionary(texts)
    print(dictionary)
    corpus_bow = [dictionary.doc2bow(text) for text in texts]
    tfidf = gensim.models.TfidfModel(corpus_bow)

    if args.save_dict:
        dictionary.save('save_models/' + args.table + '/dictionary/dictionary.dict')
        gensim.corpora.MmCorpus.serialize('save_models/' + args.table + '/dictionary/serialized_corpus.mm', corpus_bow)

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    doc_list = alldocs[:]  # For reshuffling per pass

    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))
    if args.load:
        try:
            filename = 'save_models/' + args.table + '/model_names.txt'
            print("loading store model from {}".format(filename))
            simple_models = []
            with open(filename, 'r') as model_names:
                for model in model_names:
                    simple_models.append(gensim.utils.SaveLoad.load(model[0:-1]))
        except FileNotFoundError:
            print("File not found for loading")
            simple_models = get_models(alldocs)
    else:
        simple_models = get_models(alldocs)

    if args.save:
        f = open('save_models/' + args.table + '/model_names.txt', 'w')
        for model in simple_models:
            model_name = str(model).replace('/', '-')
            save_name = 'save_models/' + args.table + '/' + model_name
            model.save(save_name)
            print(save_name)
            f.write(save_name + '\n')

        f.close()

    # The models are global in scope because of OrderedDict or ???
    best_error = run_models(simple_models, doc_list)

    results(best_error, simple_models, alldocs, args.table)

    doc_id = np.random.randint(simple_models[0].docvecs.count)  # pick random doc, re-run cell for more examples
    model = random.choice(simple_models)  # and a random model
    sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
    # model_sentiment = [int(x) for x in ]
    if args.do_plots:
        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib import pyplot as plt
        ids = [x[0] for x in sims]
        distances = [x[1] for x in sims]
        plt.plot(distances)
        plot_title = args.table + ' label:' +  str(int(labels[doc_id])) + ': \n"' + titles[doc_id] + '"'
        plt.title(plot_title)
        plt.show()

    pp()
# [x for x in zip([int(x) for x in labels], range(len(labels)))]
