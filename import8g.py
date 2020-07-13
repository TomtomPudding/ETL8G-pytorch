import struct
from PIL import Image, ImageEnhance
import glob, os, re, sys
import numpy as np

RECORD_SIZE = 8199

def read_record_ETL8G(f):
  s = f.read(8199)
  if not s: return None
  r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
  iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
  iL = iF.resize((32,32)).convert('L')
  enhancer = ImageEnhance.Brightness(iL)
  iE = enhancer.enhance(16)
  return r + (iE,)

hira_ary = np.zeros([72, 160, 32, 32], dtype=np.uint8)
kanji_ary = np.zeros([881, 160, 32, 32], dtype=np.uint8)

for j in range(1, 33):
  filename = 'ETL8G/ETL8G_{:02d}'.format(j)
  print(j, filename)
  with open(filename, 'rb') as f:
    for id_dataset in range(5):
      hira_cnt = 0
      kanji_cnt = 0
      a = 0
      for i in range(956):

        r = read_record_ETL8G(f)
        if r is None:
          break
        if b'.HIRA' in r[2]:
          hira_ary[hira_cnt, (j - 1) * 5 + id_dataset] = np.array(r[-1])
          hira_cnt += 1
        if (re.match(r"[3-4][0-9a-f]{3}", str(hex(r[1])[-4:]) )):
          kanji_ary[kanji_cnt, (j - 1) * 5 + id_dataset] = np.array(r[-1])
          kanji_cnt += 1

np.savez_compressed("hiragana.npz", hira_ary)
np.savez_compressed("kanji.npz", kanji_ary)
