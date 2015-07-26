from functools import wraps
from flask import Flask, make_response, render_template
import os

app = Flask(__name__)

def get_newest(file_mask,dir='.'):
    file_list = get_files_by_mask(file_mask,dir)
    times=[(os.path.getmtime(file_name),file_name) for file_name in file_list]
    times.sort()
    return times[-1][1]

def get_files_by_mask(name_part,dir="."):
    """return list"""
    import re
    files=[]
    all_files = os.listdir(dir)
    #reg=re.compile(regex)
    filtered = [f for f in all_files if name_part in f]
    return filtered

def add_response_headers(headers={}):
    """This decorator adds the headers passed in to the response"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            resp = make_response(f(*args, **kwargs))
            h = resp.headers
            for header, value in headers.items():
                h[header] = value
            return resp
        return decorated_function
    return decorator

def refresh(f,seconds=60):
    """This decorator passes X-Robots-Tag: noindex"""
    return add_response_headers({'meta http-equiv="refresh': str(seconds)})(f)

def refresh60(f):
    """This decorator passes X-Robots-Tag: noindex"""
    return add_response_headers({'meta http-equiv="refresh"': '60'})(f)


@app.route('/')
@refresh60
def refreshed():
    """
    This page will be served with X-Robots-Tag: noindex
    in the response headers
    """
    fname = get_newest('-generated-')
    print("Tracking",fname)
    tp.f = fname
    txt = tp.get()
    
    items = enumerate([t.replace("/"," --> ") for t in txt.split("\n") if len(t)>5][-10:]) #

    return render_template('table.html',result=items)

    return t1+table+t2


class TextProvider:
    def __init__(self,fname):
        self.f = fname

    def get(self):
        txt = "Failed to read {} :(".format(self.f)
        with open(self.f, "rb") as ff:
            txt = ff.read()
            txt = txt.decode("cp1252",errors="replace")
        return txt

tp = TextProvider(None)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)

