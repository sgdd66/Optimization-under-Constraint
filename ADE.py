# ***************************************************************************
# Copyright (c) 2018 西安交通大学
# All rights reserved
# 
# 文件名称：ADE.py
# 
# 摘    要：自适应差分进化算法,针对单目标
# 
# 创 建 者：上官栋栋
# 
# 创建日期：2018年11月27日
#
# 修改记录
# 日期  修改者   		版本     修改内容
# ------------- 		-------  ------------------------  
# ***************************************************************************


#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np
import DOE as DOE
import matplotlib.pyplot as plt

class Ind(object):
    """
    个体初始化，由进化算法采用拉丁超立方方法生成样本，通过Location来指定个体。\n
    input:\n
    location : 个体所对应的样本值\n
    output:\n
    None
    """
    def __init__(self,location):
        
        self.x=location
        #个体缩放比例因子
        self.K=np.random.uniform(0,1.2)

    def getValue(self,environment):
        '''计算个体的适应度\n
        input:\n
        envirtonment : 函数指针，进化算法所需求解的目标函数\n
        output:\n
        个体适应度存储在self.y
        '''

        self.y=environment(self.x)

    def __cmp__(self, other):
        if (self.y > other.y):
            return 1
        elif (self.y < other.y):
            return -1
        elif (self.y == other.y):
            return 0

    def __gt__(self, other):
        if (self.y > other.y):
            return True
        else:
            return False

    def __lt__(self, other):
        if (self.y < other.y):
            return True
        else:
            return False

    def __eq__(self, other):
        if (self.y == other.y):
            return True
        else:
            return False



class ADE(object):
    """自适应差分进化算法\n
    input:\n
    min,max : 向量，向量维度是搜索空间的维度，用以确定搜索空间的范围\n
    population : 种群数量\n
    CR : 0~1，父代个体与子代个体交换基因片段的概率，越大说明子代的基因片段占比越大\n
    environment : 函数指针，用以表示环境，检验个体的性能\n
    isMin : 用以确定是不是求最小值，默认为TRUE。若为FALSE表示求取最大值\n
    output:\n
    None
    """    
    def __init__(self,min,max,population,CR,environment,isMin=True):
        samples=DOE.LatinHypercube(dimension=min.shape[0],num=population,min=min,max=max)
        self.Inds=[]
        for i in range(population):
            ind=Ind(location=samples.realSamples[i,:])
            ind.getValue(environment)
            self.Inds.append(ind)
        self.environment=environment
        self.isMin=isMin
        self.CR=CR
        self.min=min
        self.max=max
        #Pm是变异率，eta是分布指数，这两个参数是多项式变异环节使用的参数
        self.Pm=1/(min.shape[0])
        self.eta=20

    def aberrance(self):
        """变异"""
        nextInds=[]
        population=len(self.Inds)
        for i in range(population):
            isOK=True
            while isOK:
                r1=np.random.randint(0,population)
                if r1==i:
                    continue
                r2=np.random.randint(0,population)
                if r1==r2 or r2==i:
                    continue
                r3=np.random.randint(0,population)
                if r3==r1 or r3==r2 or r3==i:
                    continue
                isOK=False
            #K为缩放比例因子
            K=(self.Inds[i].K+self.Inds[r1].K+self.Inds[r2].K+self.Inds[r3].K)/4
            K=K*np.exp(-self.maxGen/3*np.abs(np.random.normal()))
            x=self.Inds[r1].x+K*(self.Inds[r2].x-self.Inds[r3].x)
            ind=Ind(location=x)
            ind.K=K
            nextInds.append(ind)
        self.nextInds=nextInds

    def exchange(self):
        """交换，父代个体与子代个体按照对应索引号配对，按照概率交换部分样本值的片段，重新构成子代个体\n"""
        CR = self.CR
        inds1=self.Inds
        inds2=self.nextInds
        inds=[]
        population=len(inds1)
        dimension=inds1[0].x.shape[0]
        for i in range(population):
            x1=inds1[i].x
            x2=inds2[i].x
            x=np.zeros(dimension)
            randr = np.random.randint(0, dimension)
            for j in range(dimension):
                randb=np.random.uniform()
                if randb<=CR or j==randr:
                    x[j]=x2[j]
                else:
                    x[j]=x1[j]
            ind=Ind(location=x)
            ind.K=(inds1[i].K+inds2[i].K)/2

            #多项式变异
            for j in range(dimension):
                randk=np.random.uniform()
                if randk<0.5:
                    theta=(2*randk)**(1/(self.eta+1))-1
                else:
                    theta=1-(2-2*randk)**(1/(self.eta+1))
                randm=np.random.uniform()
                if randm<self.Pm:
                    ind.x[j]=ind.x[j]+theta*(self.max[j]-self.min[j])



            for j in range(dimension):
                if(ind.x[j]<self.min[j]):
                    ind.x[j]=self.min[j]
                if ind.x[j]>self.max[j]:
                    ind.x[j]=self.max[j]

            inds.append(ind)
        self.nextInds=inds

    def select(self):
        '''选择，从父代个体和子代个体中选择适应度最高的的个体'''
        inds=[]
        population=len(self.nextInds)
        
        for i in range(population):
            self.nextInds[i].getValue(self.environment)
            if self.isMin:
                if self.nextInds[i].y>self.Inds[i].y:
                    inds.append(self.Inds[i])
                else:
                    inds.append(self.nextInds[i])
            else:
                if self.nextInds[i].y<self.Inds[i].y:
                    inds.append(self.Inds[i])
                else:
                    inds.append(self.nextInds[i])
        self.Inds=inds

    def getProportion(self):
        '''计算最优个体的占比'''
        if self.isMin:
            self.Inds.sort()
        else:
            self.Inds.sort(reverse=True)
        num=1
        population=len(self.Inds)

        for i in range(1,population):
            if (self.Inds[i].x!=self.Inds[0].x).any():
                break
            num+=1
        return num/population

    def evolution(self,maxGen=100,maxProportion=0.8):
        '''差分进化算法\n
        input:\n
        maxGen : 最大迭代次数，当迭代次数超过maxGen，计算终止\n
        maxProportion : 最大占比，当最优个体种群占比超过maxProportion计算终止

        output:\n
        最优子代个体
        '''
        self.maxGen=maxGen
        ratio=0
        gen=0
        while gen<maxGen and ratio<maxProportion:
            self.aberrance()
            self.exchange()
            self.select()
            ratio=self.getProportion()
            print('进化代数{0}，最优值{1}，最优点{2}，最优值占比{3}'.format(gen,self.Inds[0].y,self.Inds[0].x,ratio))
            gen+=1
        return self.Inds[0]

    def saveArg(self,path=None):
        '''存储训练参数，方便继续训练\n
        input : \n
        path : 存储路径，若为空使用当前时间命名文件\n
        output: \n
        None'''
        #获取当前时间作为文件名
        if path is None:
            import time
            timemark = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            path = './Data/ADE_Population_Argument_{0}.txt'.format(timemark)
        with open(path,'w') as file:
            file.write('NUM:%d\n'%len(self.Inds))
            file.write('DIM:%d\n'%(len(self.Inds[0].x)))    
            for i in range(len(self.Inds)):
                for j in range(len(self.Inds[i].x)):
                    file.write("%.18e,"%self.Inds[i].x[j])
                file.write("%.18e,"%self.Inds[i].y)
                file.write("%.18e\n"%self.Inds[i].K)     

    def retrain(self,path,maxGen=100,maxProportion=0.8):
        '''读取文件重新训练\n
        input:\n
        path : 种群数据文件
        maxGen : 最大迭代次数，当迭代次数超过maxGen，计算终止\n
        maxProportion : 最大占比，当最优个体种群占比超过maxProportion计算终止

        output:\n
        最优子代个体
        '''
        import re
        Inds = []
        with open(path,'r') as file:
            texts = file.readlines()
            reg_int = re.compile(r'-?\d+')
            reg_float = re.compile(r'-?\d\.\d+e[\+,-]\d+')      

            num = int(reg_int.search(texts[0]).group(0))
            dim = int(reg_int.search(texts[1]).group(0))
            for i in range(num):
                text_list = reg_float.findall(texts[2+i])
                x = np.zeros(dim)
                for j in range(dim):
                    x[j] = float(text_list[j])
                
                ind=Ind(location=x)
                ind.y = float(text_list[dim])
                ind.K = float(text_list[dim+1])
                Inds.append(ind)
        self.Inds = Inds
        return self.evolution(maxGen,maxProportion)
        
if __name__=='__main__':
    # def func(X):
    #     x=X[0]
    #     y=X[1]
    #     return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-1/3*np.exp(-(x+1)**2-y**2)

    # Brain函数
    # def func(x):
    #     pi=3.1415926
    #     y=x[1]-(5*x[0]**2)/(4*pi**2)+5*x[0]/pi-6
    #     y=y**2
    #     y+=10*(1-1/(8*pi))*np.cos(x[0])+10
    #     return y      

    def func(x):
        y = (1.5-x[0]*(1-x[1]))**2+(2.25-x[0]*(1-x[1]**2))**2+(2.625-x[0]*(1-x[1]**3))**2 
        return y
    min=np.array([-3,-3])
    max=np.array([3,3])
    test=ADE(min,max,100,0.5,func,True)
    ind=test.evolution(maxGen=50)
    test.saveArg('./Data/ADE.txt')
    ind = test.retrain('./Data/ADE.txt',maxGen=500)



    