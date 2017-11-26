import requests
import xmltodict
import json
import time
from db import psqlServer

LG = 'cat:cs.LG'
AI = 'cat:cs.AI'
RESULTLIM = 'max_results='
START = '&start='

url_base_query = 'http://export.arxiv.org/api/query?search_query='

def arxiv_query(cat, num_results, start=0):
    query = url_base_query + cat + '&' + RESULTLIM + str(num_results) + START + str(start) + '&sortBy=submittedDate&sortOrder=ascending'
    r = requests.get(query)
    if r.status_code != 200:
        print("non-200 status code")
        import pdb; pdb.set_trace()
    xml = xmltodict.parse(r.text)
    jj = json.loads(json.dumps(xml))
    the_articles = jj['feed']['entry']
    if type(the_articles) == type(dict()):
        the_articles = [the_articles]
    return the_articles

def insert(articles, server,startnum):
    keys = art.keys()

    # Get the arXiv category of the article.
    category_other = art['arxiv:primary_category']['@term']

    # Get the journal_ref datum, and the number of figs and pages
    if 'arxiv:journal_ref' in keys:
        journal_ref = art['arxiv:journal_ref']['#text']
        has_journal_ref = 1
    else:
        journal_ref = None
        has_journal_ref = 0
    # Number of pages, and number of figs
    num_pages = 0
    num_figs = 0
    if 'arxiv:comment' in keys:
        num_pages = 0
        num_figs = 0
        tt = art['arxiv:comment']['#text']
        tt = tt.upper()
        ind = tt.find('PAGES')
        kk = 1
        while kk < ind:
            try:
                num_pages = int(tt[0:kk])
            except ValueError:
                break
            kk += 1
        if len(tt) < ind + 5:
            tt = tt[ind + 5:]
        try:
            while tt[0] == ',' or tt[0] == ' ':
                tt = tt[1:]

            ind = tt.find('FIGURE')
            tt = tt[0:ind]
            kk = ind - 2
            while kk >= 0:
                try:
                    num_figs = int(tt[kk:])
                except ValueError:
                    break
                kk -= 1
        except IndexError:
            pass

    # Date published
    try:
        timestamp = art['published']
    except:
        pass

    # Title
    title = art['title']

    # Get summary length in words and characters
    summary = art['summary']
    summary_length = len(summary)
    summary_wc = sum([1 if x == ' ' else 0 for x in summary])
    summary_wc += sum([1 if x == '\n' else 0 for x in summary])

    # Get first author
    authors = art['author']
    if type(authors) is type(dict()):
        authors = [authors]

    # Number of authors
    num_auth = len(authors)
    first_auth = authors[0]['name']

    statement = "INSERT INTO arx (pub_date, title, author, abstract, num_auth, num_pages, num_figs, journal_ref, has_journal_ref, summary_length, summary_wc, category, category_other)"
    statement += " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
    sv.execute_insert(statement, timestamp, title, first_auth, summary, num_auth, num_pages, num_figs, journal_ref,
                      has_journal_ref, summary_length, summary_wc, primary_category, category_other)

    # Do some logging
    print("<{}> pubdate:{} title:{} author:{}".format(startnum, timestamp, title, first_auth))


if __name__ == '__main__':
    primary_category = AI.split(':')[1]
    sv = psqlServer()

    for jj in range(12000, 13000, 1000):
        the_articles = arxiv_query(cat=LG, num_results=1000, start=jj)
        for art in the_articles:
            insert(art, sv, jj)
        time.sleep(10)

    print("done")
