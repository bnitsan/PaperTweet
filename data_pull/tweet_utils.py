import urllib
import re
import tweepy
import urlexpander
import requests

def link_entities_to_clean_arxiv_list(tweet_urls_dict):
    url_list = []
    for url_dict_i in tweet_urls_dict:
        if 'unwound_url' in url_dict_i.keys():
            url_i = url_dict_i['unwound_url']
        else:
            url_i = url_dict_i['expanded_url']
        
        # meaning link is actually a photo
        if '/photo/1' in url_i:
            #print('Early giving up on likely picture:' + url_i) # spamming too much
            continue

        # try first to expand the url
        if 'arxiv.org/' not in url_i: 
            url_i = urlexpander.expand(url_i)
            first_QM = url_i.find('?')
            if first_QM != -1:
                url_i = url_i.replace(url_i[first_QM:],'') # sometimes the url can come with a query in the end XXX?q=YYY
        
        if 'goo.gl' in url_dict_i['expanded_url']: # goo.gl are less submissive to other algorithms; this usually works
            try:
                response = requests.get(url_dict_i['expanded_url'])
                url_i = 'http' + response.url.split('&')[0].split('http')[-1]
            except:
                print('no harm done')
        
        if 'arxiv.org/' in url_i:
            if 'pdf' in url_i:
                arxiv_num_str = re.search('\d+.\d+',url_i).group()
                url_i = 'https://arxiv.org/abs/'+arxiv_num_str
            if 'lanl.' in url_i: # sometimes you find lanl.arxiv.org/...
                url_i = url_i.replace('lanl.','')
            url_list.append(url_i)
        else:
            print('Gave up on ' + url_i)

    return url_list


# based on https://stackoverflow.com/questions/42013072/extracting-external-links-from-tweets-in-python
# Doesn't seem to work better than urlexpander package
def link_expand_method_2(url):
    try:
        opener = urllib.request.build_opener()
        request = urllib.request.Request(url)
        response = opener.open(request)
        expand_url = response.geturl()
    except:
        expand_url = url
    return expand_url





def tweets_to_df(tweets, add_threads_flag=True):
    '''
    Converts "tweets", a structure obtained by client.search...() into a DataFrame of tweets ids, text, links, author ids.
    if add
    '''
    tweetIDs = []
    texts = []
    links = []
    authorID = []

    for tweet in tweets.data:
        authorID.append(tweet.author_id)

        tweetIDs.append(tweet['id'])

        if add_threads_flag:
            query = 'conversation_id: ' + str(tweet['id']) + ' from:' + str(tweet.author_id)
            thread = client.search_recent_tweets(query=query)

            thread_text = ''
            if thread.data != None:
                for tweet_thread in thread.data:
                    thread_text = tweet_thread['text'] + thread_text
                texts.append(tweet['text'] + thread_text)
            else:
                texts.append(tweet['text'])
        else:
            texts.append(tweet['text'])

        tweet_urls = tweet_utils.get_urls_single_tweet(tweet)
        links.append(tweet_urls)

    texts = [text.replace('\n',' ') for text in texts]