class Vectorizer:
    def __init__(self, fulltext, cutoff=None):
        self.vector_len=0
        self.dictionary=[]
        self._new_feed(fulltext,cutoff)

    def detokenize(self, token):
        return ""+token

    def tokenize(self,stream):
        tokens=[]
        for n in range(0,len(stream)-1,1):
            tokens.append(stream[n])
        return tokens

    def _c(self,item):
        """Will re-encode text to avoid unicode errors"""
        if type(item)==type(u" "):
            return item.encode('utf-8','replace')
        return unicode(item,errors='replace')

    def _feed(self,text):
        self.dictionary = set()
        for item in self.tokenize(text):
            self.dictionary.add(item)
        self.dictionary = [x for x in self.dictionary]
        self.vector_len = len(self.dictionary)

    def _new_feed(self,text,cutoff=None):
        dic = {}
        for item in self.tokenize(text):
            dic[item] = dic.get(item,0)+1
            if len(dic.values()) > cutoff:
                break
        dic[""]=999999
        # now sort
        self.items = list(dic.items())
        print("ITEMS",self.items)
        self.items.sort(key=lambda k: -k[1])
        #if cutoff:
        #    self.items = self.items[:cutoff]
        self.dictionary = [x for x,count in self.items]
        print("DIC",self.dictionary)
        self.vector_len = len(self.dictionary)

    def index(self,item):
        if self.dictionary.count(item):
            return self.dictionary.index(item)
        else:
            return 0

    def item(self,index,unknown_token_value=None):
        try:
            return self.dictionary[index]
        except IndexError:
            return unknown_token_value

    def len(self):
        return len(self.dictionary)

    def vector(self,item):
        v = [0.0] * self.len()
        v[self.index(item)] = 1.0
        return v

    def to_matrix(self,item_list):
        """Generate a one-hot matrix from a sequence of items"""
        mat=[]
        for item in self.tokenize(item_list):
            mat.append(self.vector(item))
        return mat

    def from_vector(self,vec,unknown_token_value=None):
        winner = max(vec)
        return self.detokenize(self.item(vec.index(winner)))

    def from_vector_rand(self,vec,randomization=0.5,unknown_token_value=None):
        from random import random
        srt = [(v,x) for x,v in enumerate(vec)]
        srt.sort()
        srt.reverse()
        for winner, idx in srt:
            if random()<(1-randomization):
                break
        return self.detokenize(self.item(idx,unknown_token_value))



def testdata():
    return """topkek. Was ist an dem Artikel nicht bescheuert? . . . Was unterscheidet die bewaffneten Gruppen des Staates dann noch von den bewaffneten Gruppen der Kartelle? . Mord bleibt Mord phrasen bleiben phrasenDann lege den Unterschied doch mal dar.  . Und nun?Da hast du deinen unterschied. . meine Fresse. Den Tag muss ich mir im Kalender markieren.  Huh?Das wir mal einer Meinung sind.  Deine Flagge deutet auf eine Tendenz zum Anarchosyndikalismus hin.  Ich bin wertekonservativ mit einem Hang zum Anarchokapitalismus.  Es gibt also eine gewisse Schnittmenge an Meinungen :)In der Tat.  Besonders viertel, wo sie mit BW13-Tattoos rumlaufen, sollte man meiden.  Das sind fiese Gangster. An der Aussage kann man ja sehen, das du noch nie dort warst. . . . . . . Die Rede von Frau Merkel davor:
https://www. youtube. com/watch?v=lYX37ZljEto

Die Rede von Herrn Gabriel danach:
https://www. youtube. com/watch?v=6E6-mV0jJAAdie Antwort vom Gabriel war schon nicht schlecht!Ist ja in Ordnung, wenn das der Snowdens Edward macht, aber wehe dem, der das in der Bundesrepublik versucht. . . . . . . http://i. imgur. com/YCzMI. gifFind ich gut.  Hoffentlich kommt der Netzpolitik-Hampel in den Knast. . . . . . . . Was haben die Deutschen eigentlich nicht erfunden?Ricola. ich mag seine passiv aggresive art nicht. . Immer diese Leute die nicht verstehen was ein x-post ist. Bild oder es ist nicht passiert!Immer diese Leute die nicht verstehen was ein x-post ist. Also ich dachte zumindest du solltest wissen was ein x-post ist. [Well then](http://www. allmystery. de/i/t330413_unexpected-monk-meme-generator-do-me-a-f. jpg?bc). . . . Nein, Skrotum ist der Hodensack. Immer diese Leute die nicht verstehen was ein x-post ist. . Ist das jetzt gut, weil die AfD endlich als rechts-konvervative Partei bekannt wird, oder schlecht, weil sie endlich als rechts-konvervative Partei bekannt wird ?. Hat sich aber offiziell gerne als "liberal" und Alternative zur FDP dargestellst . . .  damit isses wohl nun zu Ende.  . . . . . . . Die AfD ist die vorzeitige  Nachfolgerpartei der NPD. . . . . . . .  oder auch der politische Arm der Pegida. . . . Schill.  [Vielleicht brauchen wird bald ein neues Lied](https://www. youtube. com/watch?v=bAl5HbBJe98). [Der braune PEGIDA-Dreckmob hat schon eins](https://soundcloud. com/yellowumbrella2/no-pegida-yellow-umbrella-ronny-trettmann-tiny-dawson). Ich ich auche Kacke.  Ich fand die AfD schon in vielen Punkten in Ordnung aber jetzt driften sie mir zu weit Rechts.

Es fehlt einfach eine ordentliche rechts-liberale Partie in Deutschland. Es gab da ja mal die FDP, aber deren Rolle hat jetzt ja schon die SPD eingenommen. . Bernd Lucke for President!!Jetzt mach mer mal langsam. &gt;Die Halle tobt, als Petry zum Finale ansetzt: Die Links-rechts-Einordnung vergifte die Demokratie im Land.
"""

class VectorizerTwoChars(Vectorizer):

    def tokenize(self,stream):
        return self.tokenize_2lemma(stream)

    def detokenize(self,token):
        return self.detokenize_2lemma(token)

    def detokenize_2lemma(self, token):
        if len(token)>1:
            a,b=token
            return a+b
        elif len(token):
            return a
        else:
            return None

    def tokenize_2lemma(self,stream):
        tokens=[]
        for n in range(0,len(stream)-1,2):
            tokens.append((self._c(stream[n]),self._c(stream[n+1])))
        return tokens

    def _c(self,item):
        return item

def test():
    #reload(vectorizer)
    d=testdata()
    v=Vectorizer(d)
    print("Len of dictionary:",v.len())

    toks = v.tokenize("heute")
    print(toks)

    mat = v.to_matrix("heute aber nicht")
    print(mat)

    text = [v.detokenize(v.from_vector(x)) for x in  mat]
    print("".join(text))
