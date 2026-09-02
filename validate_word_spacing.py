from math import sin, pi

SR=48_000
EXTRA_MS=30
MIN_MS=8
MAX_MS=120
MIN_VOICED_MS=35

class Pacer:
    def __init__(self, extra_ms=EXTRA_MS):
        ms=lambda x: SR*x//1000
        self.extra=ms(extra_ms); self.min_gap=ms(MIN_MS); self.max_gap=ms(MAX_MS); self.min_voiced=ms(MIN_VOICED_MS)
        self.env=0.0; self.peak=0.0; self.heard=False; self.quiet=0; self.voiced=0
    def push(self, xs):
        if self.extra==0: return list(xs)
        out=[]
        for x in xs:
            a=abs(x); self.env=self.env*.995+a*.005; self.peak=max(self.peak*.99995,self.env)
            threshold=min(.018,max(.0015,self.peak*.10))
            is_quiet=self.heard and self.env<threshold
            if is_quiet:
                self.quiet+=1; out.append(x); continue
            if self.quiet:
                word=self.min_gap<=self.quiet<=self.max_gap and self.voiced>=self.min_voiced
                if word:
                    out.extend([0.0]*self.extra); self.voiced=0
                self.quiet=0
            out.append(x)
            if self.env>=threshold:
                self.heard=True; self.voiced+=1
        return out

def tone(ms, amp=.06, hz=180):
    n=SR*ms//1000
    return [amp*sin(2*pi*hz*i/SR) for i in range(n)]
def silence(ms): return [0.0]*(SR*ms//1000)

# A natural short inter-word gap should gain approximately the requested 30 ms.
x=tone(100)+silence(40)+tone(100)
p=Pacer(); y=p.push(x)
assert len(y)==len(x)+SR*EXTRA_MS//1000, (len(x),len(y))

# Off is bit-for-bit length-preserving.
p=Pacer(0); assert p.push(x)==x

# Long punctuation pause stays unchanged.
x2=tone(100)+silence(200)+tone(100)
p=Pacer(); assert len(p.push(x2))==len(x2)

# Trailing silence is not expanded.
x3=tone(100)+silence(40)
p=Pacer(); assert len(p.push(x3))==len(x3)

# State carried across a chunk boundary produces the same length as one-shot processing.
x4=tone(100)+silence(40)+tone(100)
split=SR*115//1000
p=Pacer(); streamed=p.push(x4[:split])+p.push(x4[split:])
p=Pacer(); whole=p.push(x4)
assert len(streamed)==len(whole)==len(x4)+SR*EXTRA_MS//1000

print("word-spacing pacing validation OK")
