import praw
from random import randint

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
        txt += "#_BEGIN_# " + c.body + " #_END_# \n\n"

    if REDDIT_MODE == "TEXT":
        ## TODO ## change this to html.encode!!
        ## TODO ## change this to html.encode!!
        ## TODO ## change this to html.encode!!
        ## TODO ## change this to html.encode!!
        if type(txt) == str: # python3
            #txt = txt.encode('ascii','xmlcharrefreplace').decode()
            txt = txt.encode('ascii','xmlcharrefreplace').decode()
        else:
            txt = txt.encode('utf-8', errors='xmlcharrefreplace')
        return txt
    elif REDDIT_MODE == "WORDS":
        for item in "*":
            txt=txt.replace(item,"")
        for item in "\n,;":
            txt=txt.replace(item," "+item+" ")
        for item in "!?.:":
            txt=txt.replace(item+" "," "+item+" ")

        ## TODO : Test!!
        uni = str(txt.encode('utf-8', errors='replace'))
        ## TODO might have to uni = str(uni)
        return [x+" " for x in uni.split(" ")]
    else:
        raise ValueError("Need to specify REDDIT_MODE as 'TEXT' or 'WORDS'")
