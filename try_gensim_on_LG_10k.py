
# coding: utf-8



import locale

import numpy as np

from db import psqlServer
sv = psqlServer()

control_chars = [chr(0x85)]
locale.setlocale(locale.LC_ALL, 'C')





def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text




AI = "arx_AI"        # artificial intelligence
FA = "arx_math_FA"   # functional analysis
GR = "arx_GRQC"      # general relativity and quantum cosmology
LG = "arx"           # learning
NT = "arx_math_NT"   # number theory




# How many items in all tables?
UNION = " UNION ALL "
SEL_C = "SELECT COUNT(*) as FOO FROM "
sum_statement = "SELECT SUM(FOO) FROM ("
sum_statement += SEL_C + AI
sum_statement += UNION + SEL_C + FA 
sum_statement += UNION + SEL_C + GR 
sum_statement += UNION + SEL_C + LG 
sum_statement += UNION + SEL_C + NT
sum_statement += ") as B;"

total = sv.execute(sum_statement)[0]['sum']
print(total)




# Per table
SEL_C = "SELECT COUNT(*) FROM "
total_AI = sv.execute(SEL_C + AI)[0]['count']
total_FA = sv.execute(SEL_C + FA)[0]['count']
total_GR = sv.execute(SEL_C + GR)[0]['count']
total_LG = sv.execute(SEL_C + LG)[0]['count']
total_NT = sv.execute(SEL_C + NT)[0]['count']

print("AI:{}".format(total_AI))
print("FA:{}".format(total_FA))
print("GR:{}".format(total_GR))
print("LG:{}".format(total_LG))
print("NT:{}".format(total_NT))




# Get some rows from the AI table
rows = sv.execute("SELECT * FROM arx LIMIT 10000")
corpus = [x['abstract'] for x in rows]
labels = [float(x['has_journal_ref']) for x in rows]

#rows2 = sv.execute("SELECT abstract FROM arx_math_FA LIMIT 10000")
#corpus += [x['abstract'] for x in rows2]
#labels += [x['has_journal_ref'] for x in rows2]

#rows3 = sv.execute("SELECT abstract FROM arx_GRQC LIMIT 30000")
#corpus += [x['abstract'] for x in rows3]
#labels += [x['has_journal_ref'] for x in rows2]




corpus_n = []
jj = 0
for corp in corpus:
    corpus_n.append(u"_*{0} {1}\n".format(jj, normalize_text(corp)))
    jj += 1




len(corpus_n)




import gensim
from collections import namedtuple




SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
alldocs = []
for line_no, line in enumerate(corpus_n[0:2500]):
    tokens = gensim.utils.to_unicode(corpus_n[line_no]).split()
    words = tokens[1:]
    tags = [line_no]
    split = 'train'
    #split = ['train', 'test', 'extra', 'extra'][line_no//12500]
    sentiment = labels[line_no]
    #sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//6750]
    alldocs.append(SentimentDocument(words, tags, split, sentiment))

for line_no, line in enumerate(corpus_n[2500:5000]):
    tokens = gensim.utils.to_unicode(corpus_n[line_no]).split()
    words = tokens[1:]
    tags = [line_no]
    split = 'test'
    #split = ['train', 'test', 'extra', 'extra'][line_no//12500]
    sentiment = labels[line_no]
    #sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//6750]
    alldocs.append(SentimentDocument(words, tags, split, sentiment))
    
for line_no, line in enumerate(corpus_n[5000:]):
    tokens = gensim.utils.to_unicode(corpus_n[line_no]).split()
    words = tokens[1:]
    tags = [line_no]
    split = 'extra'
    #split = ['train', 'test', 'extra', 'extra'][line_no//12500]
    sentiment = None
    #sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//6750]
    alldocs.append(SentimentDocument(words, tags, split, sentiment))    




len(alldocs)




train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # For reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))




from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing




cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"




simple_models = [
    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/ average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
]




simple_models[0].build_vocab(alldocs)




print(simple_models[0])
for model in simple_models[1:]:
    """Reuse shareable structures from other_model."""
    model.reset_from(simple_models[0])
    print(model)




models_by_name = OrderedDict((str(model), model) for model in simple_models)
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])




models_by_name





# Predictive Evaluation Methods

# Let's define some helper methods for evaluating the performance of our Doc2vec
# using paragraph vectors. We will classify document sentiments using a logistic 
# regression model based on our paragraph embeddings. We will compare the error 
# rates based on word embeddings from our various Doc2vec models.
import numpy as np
import statsmodels.api as sm
from random import sample

# For timing
from contextlib import contextmanager
from timeit import default_timer
import time 




@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    
def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
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





# Bulk Training

# We use an explicit multiple-pass, alpha-reduction approach as sketched in 
# this gensim doc2vec blog post with added shuffling of corpus on each pass.
# https://rare-technologies.com/doc2vec-tutorial/
from collections import defaultdict
best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved









from random import shuffle
import datetime

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())

for epoch in range(passes):
    shuffle(doc_list)  # Shuffling gets best results
    
    for name, train_model in models_by_name.items():
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
        if err <= best_error[name]:
            best_error[name] = err
            best_indicator = '*' 
        print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

        if ((epoch + 1) % 5) == 0 or epoch == 0:
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                infer_err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs, infer=True)
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if infer_err < best_error[name + '_inferred']:
                best_error[name + '_inferred'] = infer_err
                best_indicator = '*'
            print("%s%f : %i passes : %s %ss %ss" % (best_indicator, infer_err, epoch + 1, name + '_inferred', duration, eval_duration))

    print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta
    
print("END %s" % str(datetime.datetime.now()))




# Print best error rates achieved
print("Err rate Model")
for rate, name in sorted((rate, name) for name, rate in best_error.items()):
    print("%f %s" % (rate, name))




doc_id = np.random.randint(simple_models[0].docvecs.count)  # Pick random doc; re-run cell for more examples
print('for doc %d...' % doc_id)
for model in simple_models:
    inferred_docvec = model.infer_vector(alldocs[doc_id].words)
    print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))





import random

doc_id = np.random.randint(simple_models[0].docvecs.count)  # pick random doc, re-run cell for more examples
model = random.choice(simple_models)  # and a random model
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
print(u'TARGET (%d): «%s»\n' % (doc_id, ' '.join(alldocs[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(alldocs[sims[index][0]].words)))


