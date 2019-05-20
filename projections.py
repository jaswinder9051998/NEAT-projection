import random

import numpy as np
from scipy.interpolate import interp1d
import warnings
import time
import sys
import copy
from p5 import *
import math
warnings.simplefilter("ignore")

LEN=1800
BRE=700
WHITE=(255,255,255)
BLUE=(0,0,255)
RED=(255,0,0)
GREEN=(0,255,0)
BOUN=(165,42,42)
PINK=(255,192,203)

maplogicV = interp1d([0,1],[1,20])
maplogicA = interp1d([0,1],[10,170])
class target:
    def __init__(self,x):
        self.x=x
        
class ball:
    def __init__(self,x,y):
        self.dir=0
        self.x=x
        self.y=y
        self.velocity=0
        self.colour=BLUE
        self.state=True
        self.size=10
        self.fitness=0.0
        self.life=True
    def initiate(self,v):
        self.dir=maplogicA(v[0])
        self.velocity=maplogicV(v[1])
    def fitnesss(self,xx):
        if self.x==xx:
            self.fitness=100000
            return 100000
        else:
            self.fitness=1/((self.x-xx)*(self.x-xx))
        return self.fitness
    def move(self,t):
        gravity=9.81
        xinc=self.velocity*math.cos(math.pi*(self.dir)/180)*t
        self.x=self.x+self.velocity*math.cos(math.pi*(self.dir)/180)*t
        self.y=self.y-(self.velocity * math.sin(math.pi*(self.dir)/180)*t - (gravity/2)*t*t)
        if self.x<=LEN/4 and self.x+xinc>=LEN/4 and self.y>=BRE-80 and self.y<=BRE:
            self.life=False
            self.state=False
        if(self.y>BRE):
            self.y=BRE
            self.state=False
        
        



class FFSN_MultiClass:
    def __init__(self, n_inputs, n_outputs, hidden_sizes=[3]):
        
        self.nx = n_inputs
        self.ny = n_outputs
        self.nh = len(hidden_sizes)
        self.sizes = [self.nx] + hidden_sizes + [self.ny] 

        self.W = {}
        self.B = {}
        for i in range(self.nh+1):
            self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
            self.B[i+1] = np.random.randn(1,self.sizes[i+1] )
     
          
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
  
    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh):
            self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]
            self.H[i+1] = self.sigmoid(self.A[i+1])
        self.A[self.nh+1] = np.matmul(self.H[self.nh], self.W[self.nh+1]) + self.B[self.nh+1]
        self.H[self.nh+1] = self.softmax(self.A[self.nh+1])
        return self.H[self.nh+1]
  
    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()
 
    def grad_sigmoid(self, x):
        return x*(1-x)
    
    def cross_entropy(self,label,pred):
        yl=np.multiply(pred,label)
        yl=yl[yl!=0]
        yl=-np.log(yl)
        yl=np.mean(yl)
        return yl

class population:
    def __init__(self,pops,mutR):
        self.pops=pops
        self.mutR=mutR
        self.popNN=[]
        self.balls=[]
        self.fit=[]
        self.matingp=[]
    def create(self,):
        self.balls=[]
        self.popNN=[]
        for i in range(self.pops):
            self.balls.append(ball(30,BRE))
        for i in range(self.pops):
            self.popNN.append(FFSN_MultiClass(2,2,[6,4,6]))
        for ballz,i in zip(self.balls,range(self.pops)):
            ballz.initiate(self.popNN[i].forward_pass(np.asarray([30,LEN/2]))[0])
    def fitness(self,):
        self.fit=[]
        for b in self.balls:
            self.fit.append(b.fitnesss(LEN/2))
        

    def reset(self,):
        for b in self.balls:
            b.x=30
            b.y=BRE-10
            for ballz,i in zip(self.balls,range(self.pops)):
                ballz.initiate(self.popNN[i].forward_pass(np.asarray([30,LEN/2]))[0])
                ballz.state=True
                ballz.life=True
            

    def mating(self,):
        self.fitness()
        self.matingp=[]
        maxx=max(self.fit)
        for i in range(self.pops):
            n=(int)((self.fit[i])/(maxx)*100)
            for j in range(n):
                self.matingp.append(self.popNN[i])
        #print(len(self.matingp))
            
            
        
        
    def reprod(self,):
        self.mating()
        for i in range(self.pops):
            m=random.randint(0,len(self.matingp)-1)
            self.popNN[i].W=copy.deepcopy(self.matingp[m].W)
            self.popNN[i].B=copy.deepcopy(self.matingp[m].B)
            for j in range(len(self.popNN[i].W)):
                for k in range(len(self.popNN[i].W[j+1])):
                    for l in range(len(self.popNN[i].W[j+1][k])):
                        if(random.random()<self.mutR):
                            self.popNN[i].W[j+1][k][l]=random.random()
            for j in range(len(self.popNN[i].B)):
                for k in range(len(self.popNN[i].B[j+1][0])):
                    
                    if(random.random()<self.mutR):
                        self.popNN[i].B[j+1][0][k]=random.random()
                    
                    
            

            








gravity=9.81

t=0
turn=0
ind=0
pp=population(100,0.001)
WW={}
pp.create()
SIZE=[]

def setup():
    size(LEN,BRE)
    
    

def draw():
    global  gravity,t,pp,turn,ind,WW,SIZE
    
    background(10,55,80)
    fill(161, 181, 108)
    stroke(300)
    t +=0.02
    for b in pp.balls:
        if b.state is True:
            b.move(t)

    for oo in pp.balls:
        circle((oo.x,oo.y),oo.size)
    circle((LEN/2,BRE),50)
    line((LEN/2,0),(LEN/2,BRE))
    fill(1,200,60)
    line((LEN/4,BRE-80),(LEN/4,BRE))
    pp.fitness()
    
        
                            
        
        
    if(t>=10):
        pp.fitness()
        
        pp.reprod()
        m=pp.fit.index(max(pp.fit))
        print(pp.popNN[m].W)
        
        
        
        pp.reset()
        t=0
        turn+=1



run(frame_rate=600)
