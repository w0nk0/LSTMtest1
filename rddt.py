import html
import praw
from random import randint

BEGIN_TOKEN = '#_B_#'
END_TOKEN = '#_E_#'
MAX_POSTS = 50
DEBUG = False

class FlatSubreddit:
    def __init__(self, subreddit, max_posts = MAX_POSTS, cache=True, escape_func=html.escape ):
        self.sub = subreddit
        self.max_posts = max_posts
        self._text = ""
        self._cache = cache
        self.escape_func = escape_func

    def text(self, verbose=False):
        if not self._text:
            self._flatten(verbose)
        return self._text or "."

    def _make_cache_name(self):
        return "rddt-%s-%d.cache" % (self.sub, self.max_posts)

    def _flatten(self, verbose=False):
        if self._cache:
            try:
                with open(self._make_cache_name(),"rt") as f:
                    self._text = f.read()
                if len(self._text):
                    return self._text
            except:
                self._text = ""

        print("Reading %s.." % self.sub,end="")
        result = " "
        sub = self.sub
        bot = self.r = praw.Reddit("Dada reader")
        subreddit = bot.get_subreddit(sub)
        ctr = 0
        for post in subreddit.get_new(limit=self.max_posts):
            if verbose and (ctr % 1) == 9: print("\r {} posts read!".format(ctr+1),end="")
            ctr += 1
            if DEBUG: print(post)
            comments = self._check_comments(post)
            result += comments
        #result = result.replace(".", ". ")
        #result = self.escape_func(result)
        self._text = result
        if verbose: print(" done.")

        try:
            with open(self._make_cache_name(),"wt") as f:
                f.write(result)
        except:
            warn("Couldn't write cache file")

        return result



    def _check_comments(self,post):
        txt = ""

        submission = self.r.get_submission(submission_id = post.id)
        flat_comments = praw.helpers.flatten_tree(submission.comments)

        try:
            txt = BEGIN_TOKEN + submission.body + END_TOKEN
        except:
            body = "NoBody"

        if DEBUG: print("Parsing comments")
        for comment in flat_comments:
            if DEBUG: print(".",end="")
            try:
                txt += BEGIN_TOKEN + str(comment.body) + END_TOKEN
            except:
                txt += "."
        if DEBUG: print("\n")

        return txt

    def check_if_user_posted(self, submission, user):
        flat_comments = praw.helpers.flatten_tree(submission.comments)
        users = []
        for p in submission.comments:
            author = p.author.name
            users.append(author)
        print("DDDD   Users: ", users)
        return user in users


def redditor_text(redditor, amount=50, justonerandom=False, REDDIT_MODE='TEXT'):
    """Return string or array of string ('WORDS' MODE) from one redditor's recent comments"""
    r = praw.Reddit("LSTM test")
    me = r.get_redditor(redditor)
    # print me
    all_comments = me.get_comments(limit=amount)
    txt = ""
    if justonerandom:
        comment_no = randint(0, amount - 1)
        return [c for c in all_comments][comment_no].body
    for c in all_comments:
        txt += BEGIN_TOKEN + c.body + END_TOKEN + "\n\n"

    if REDDIT_MODE == "TEXT":
        ## TODO ## change this to html.encode!!
        ## TODO ## change this to html.encode!!
        ## TODO ## change this to html.encode!!
        ## TODO ## change this to html.encode!!
        if type(txt) == str: # python3
            #txt = txt.encode('ascii','xmlcharrefreplace').decode()
            #txt = txt.encode('ascii','xmlcharrefreplace').decode()
            pass
        else:
            #txt = txt.encode('utf-8', errors='xmlcharrefreplace')
            pass
        #txt = html.escape(txt)
        return txt
    elif REDDIT_MODE == "WORDS":
        for item in "*":
            txt=txt.replace(item,"")
        for item in "\n,;":
            txt=txt.replace(item," "+item+" ")
        for item in "!?.:":
            txt=txt.replace(item+" "," "+item+" ")

        ## TODO : Test!!
        uni = txt.encode('cp1252', errors='replace').decode('cp1252', errors='replace')
        ## TODO might have to uni = str(uni)

        return [html.escape(x+" ") for x in uni.split(" ")]
    else:
        raise ValueError("Need to specify REDDIT_MODE as 'TEXT' or 'WORDS'")
