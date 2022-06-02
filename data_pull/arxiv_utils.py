from html.parser import HTMLParser
import requests

class arXivHTMLParser(HTMLParser):
    '''
    Based on https://docs.python.org/3/library/html.parser.html
    '''
    def __init__(self):
        HTMLParser.__init__(self)
        
        self.title = ''
        self.authors = []
        self.abstract = ''
        
        self.title_flag = False
        self.authors_flag = False
        self.abstract_flag = False
        
    def handle_starttag(self, tag, attrs):
        #print("Encountered a start tag:", tag)
        if tag == 'title' and self.title == '' :
            self.title_flag = True

    def handle_endtag(self, tag):
        #print("Encountered an end tag :", tag)
        if tag == 'title':
            self.title_flag = False
            
        if self.authors_flag and tag == 'div':
            self.authors_flag = False
        
        if self.abstract_flag and tag == 'blockquote':
            self.abstract_flag = False
        
    def handle_data(self, data):
        #print("Encountered some data  :", data)
        if self.title_flag:
            self.title = data
            
        if data == 'Authors:':
            self.authors_flag = True
        
        if self.authors_flag and data != 'Authors:' and data != ', ':
            self.authors.append(data)

        if data == 'Abstract:':
            self.abstract_flag = True
            
        if data != 'Abstract:' and self.abstract_flag:
            self.abstract = self.abstract + data


            
class arXivHTMLParserOriginal(HTMLParser):
    '''
    Based on https://docs.python.org/3/library/html.parser.html
    '''
    def __init__(self):
        HTMLParser.__init__(self)
        
        self.title = ''
        self.authors = []
        self.abstract = ''
        
        self.title_flag = False
        self.authors_flag = False
        self.abstract_flag = False
        
    def handle_starttag(self, tag, attrs):
        #print("Encountered a start tag:", tag)
        if tag == 'title' and self.title == '' :
            self.title_flag = True

    def handle_endtag(self, tag):
        #print("Encountered an end tag :", tag)
        if tag == 'title':
            self.title_flag = False
            
        if self.authors_flag and tag == 'div':
            self.authors_flag = False
        
        if self.abstract_flag and tag == 'blockquote':
            self.abstract_flag = False
        
    def handle_data(self, data):
        #print("Encountered some data  :", data)
        if self.title_flag:
            self.title = data
            
        if data == 'Authors:':
            self.authors_flag = True
        
        if self.authors_flag and data != 'Authors:' and data != ', ':
            self.authors.append(data)

        if data == 'Abstract:':
            self.abstract_flag = True
            
        if data != 'Abstract:' and self.abstract_flag:
            self.abstract = self.abstract + data

        
    
def get_arXiv_details(url):
    '''
        This method uses the above HTML parser (arXivHTMLParser)
        accepts a url of the variety 'https://arxiv.org/abs/xxxx.xxxxx'
        returns the [title, authors, abstracts]
        
        does not support comments, journal, and other information -- but can be extended to support it by modifying this method and the parser
    '''
    url = url.replace('arxiv','export.arxiv') # arxiv asks bots to use this website, see https://arxiv.org/denied.html ; otherwise it blocks the bot often
    
    req = requests.get(url, 'html.parser')
    parser = arXivHTMLParser()
    parser.feed(req.text)
    
    parser.title = parser.title[13:] # in title: remove [xxxx.xxxxx] prefix
    parser.abstract = parser.abstract.replace('\n',' ')[1:] # replace \n by space and remove two spaces in the beginning ('  '). Used to be 2: with arxiv.org rather than export.arxiv.org
    
    # remove extra entries from author list. Was unnecessary in arxiv.org, became necessary with export.arxiv.org
    try:
        parser.authors.remove('\n')
    except:
        pass
    try:
        while True:
            parser.authors.remove(', \n')
    except ValueError:
        pass
    
    return parser.title, parser.authors, parser.abstract 