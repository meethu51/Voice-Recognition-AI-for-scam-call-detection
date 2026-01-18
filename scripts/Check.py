import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, soundfile as sf
from models.ecapa import ECAPA

def l2n(x): 
    n = np.linalg.norm(x) + 1e-9
    return (x / n).astype("float32")

def cos(a,b): 
    return float(np.dot(a,b) / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))

ecapa = ECAPA(device="cpu")

A = r"C:\Users\asus\Desktop\Test folder subject\260-123440-0002.wav"  # one you enrolled
B = r"C:\Users\asus\Desktop\260-123288-0002.wav"  # the one you score

w1,sr1 = sf.read(A);  w2,sr2 = sf.read(B)
if w1.ndim>1: w1 = w1.mean(axis=1)
if w2.ndim>1: w2 = w2.mean(axis=1)
assert sr1==16000 and sr2==16000, (sr1,sr2)

e1 = l2n(ecapa.embed(w1.astype("float32")))
e2 = l2n(ecapa.embed(w2.astype("float32")))
print("cos =", cos(e1, e2))