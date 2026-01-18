import numpy as np, soundfile as sf
from models.ecapa import ECAPA

def l2n(x): 
    n = np.linalg.norm(x) + 1e-9
    return (x / n).astype("float32")

def cos(a,b): 
    return float(np.dot(a,b) / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))

ecapa = ECAPA(device="cpu")   # same as app

wav1, sr1 = sf.read(r"C:\Users\asus\Desktop\Test folder subject\260-123440-0002.wav")
wav2, sr2 = sf.read(r"C:\Users\asus\Desktop\Test folder subject\260-123440-0010.wav")
if wav1.ndim>1: wav1 = wav1.mean(axis=1)
if wav2.ndim>1: wav2 = wav2.mean(axis=1)
assert sr1==16000 and sr2==16000, (sr1,sr2)  # should be 16000 mono

e1 = l2n(ecapa.embed(wav1.astype("float32")))
e2 = l2n(ecapa.embed(wav2.astype("float32")))
print("cos=", cos(e1, e2))
print("||e1||, ||e2|| =", np.linalg.norm(e1), np.linalg.norm(e2))
