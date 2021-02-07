#! /usr/bin/env python3.8
import matplotlib.pyplot as plt

colors = ['r', 'g', 'b']
styles = ['-', '--', '-.', ':']
idx_c = 0
idx_s = 0

def next_line():
  global idx_c
  global idx_s
  line = colors[idx_c]+styles[idx_s]
  idx_c = idx_c+1
  idx_s = idx_s+1
  return line

def plot(name, *data):
  plt.plot(*data, next_line(), label=name)

def show():
  plt.legend(loc='best')
  plt.show()

def save(path):
  plt.legend(loc='best')
  plt.savefig(path)

if __name__ == '__main__':
  plot('aaa', [1, 2, 3], [1, 2, 3])
  plot('abb', [1, 2, 3], [1, 4, 9])
  save('xxx.png')
