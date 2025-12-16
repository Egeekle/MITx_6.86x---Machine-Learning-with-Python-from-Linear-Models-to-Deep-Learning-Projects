import random
import numpy as np

from IPython.display import Markdown, display


def print_gato2(mat, accion=(-1,-1)):
      if accion[0]>=0:
        mat[accion[0]][accion[1]]=3
      for i in mat:
        for j in i:
          if j==0:
              print(".", end=" ")
          elif j==1:
            print("X", end=" ")
          elif j==2:
            print("O", end=" ")
          elif j==3:
            print("\x1b[1;31mO\x1b[0m", end=" ")
        print()


class gato:
    def __init__(self, Q, V, verbose=0):
      self.Q=Q
      self.V=V
      self.reset()
      self.verbose=verbose
      self.epsilon=1


    def get_possible_moves(self):
      possible=[]
      row=-1
      for i in self.mat:
        col=-1
        row+=1
        for j in i:
          col+=1
          if j==0:
            possible.append((row,col))
      return possible

    def random_move(self):
        possible=self.get_possible_moves()
        if len(possible)==0:
          return None
        bestaction=random.choice(possible)
        return bestaction

    def reset(self):
      self.mat=np.array([[0,0,0], [0,0,0],[0,0,0]])

    def testwin(self):
      #diagonales:
      res=0
      if self.mat[0][0]==self.mat[1][1]==self.mat[2][2] and self.mat[2][2]>0:
        return self.mat[2][2]
      if self.mat[0][2]==self.mat[1][1]==self.mat[2][0] and self.mat[2][0]>0:
        return self.mat[2][0]

      #verticales
      for k in range(3):
        if self.mat[0][k]==self.mat[1][k]==self.mat[2][k] and self.mat[2][k]>0:
          return self.mat[2][k]

      #horizonales
      for k in range(3):
        if self.mat[k][0]==self.mat[k][1]==self.mat[k][2] and self.mat[k][2]>0:
          return self.mat[k][2]

      return 0

    def play(self, action):
      self.mat[action]=1
      return self.testwin()

    def answer(self, return_action=False, force_action=[]):
      bestaction=None
      allactions=[]
      Q=self.Q
      if self.testwin()==0:
        bestscore=-100
        if random.random()>self.epsilon:
          actions=self.get_possible_moves()
          s= arr_to_string(self.mat)
          random.shuffle(actions)
          for a in actions:
            allactions.append( round(Q.get((s,a), 0), 3) )
            if (Q.get((s,a), 0)>bestscore):
              bestscore=Q.get((s,a), 0) # Q.get((s,a), 0)==Q[(s,a)]
              bestaction=a
        if bestaction==None:
          bestaction=self.random_move()
          if bestaction==None:
            return self.testwin()
        elif self.verbose>0:
          print("Choosing best action out of "+str(allactions))
        if len(force_action)==2:
          bestaction=force_action
        self.mat[bestaction]=2
      if return_action:
        return self.testwin(),bestaction
      return self.testwin()

    def print_gato(self):
      for i in self.mat:
        for j in i:
          if j==0:
            print(".", end=" ")
          elif j==1:
            print("X", end=" ")
          elif j==2:
            print("O", end=" ")
        print()

gat = gato({}, {})
gat.print_gato()
g=0
while (g==0) and (gat.random_move()!=None):
  if gat.random_move()!=None:
    g=gat.play(gat.random_move())
    gat.print_gato
    print()
    if gat.random_move()!=None:
      g=gat.answer()
      gat.print_gato()
      print()
g = gat.testwin()
print("Winner" + str(g))

gat = gato({}, {})
gat.print_gato()

gat.play((x,x           ))
g=gat.answer()
gat.print_gato()
if g > 0:
  print("Winner" + str(g))

