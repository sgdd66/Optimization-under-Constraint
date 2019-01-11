# ***************************************************************************
# Copyright (c) 2019 西安交通大学
# All rights reserved
# 
# 文件名称：Main.py
# 
# 摘    要：约束优化方法，该方法会应用DOE,Kriging,ADE,SVM等多种方法
#
# 创 建 者：上官栋栋
# 
# 创建日期：2019年1月10号
#
# 修改记录
# 日期  修改者   		版本     修改内容
# ------------- 		-------  ------------------------  
# ***************************************************************************

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Kriging import Kriging,writeFile
from SVM import SVM
from DOE import LatinHypercube
import numpy as np

class TestFunction_G8(object):
    '''测试函数G8:\n
    变量维度 : 2\n
    搜寻空间 : 0.001 ≤ xi ≤ 10, i = 1, 2.\n
    全局最小值 : x∗ = (1.2279713, 4.2453733) , f (x∗) = 0.095825.'''

    def __init__(self):
        '''建立目标函数和约束'''
        self.aim = lambda x:-(np.sin(2*np.pi*x[0])**3*np.sin(2*np.pi*x[1]))/(x[0]**3*(x[0]+x[1]))

        #约束函数，小于等于0为满足约束
        g1 = lambda x:x[0]**2-x[1]+1
        g2 = lambda x:1-x[0]+(x[1]-4)**2
        self.constrain = [g1,g2]

        self.dim = 2
        self.min = [0.001,0.001]
        self.max = [10,10]

        self.optimum = [1.2279713, 4.2453733]

    def isOK(self,x):
        '''检查样本点x是否违背约束，是返回-1，否返回1\n
        input : \n
        x : 样本点，一维向量\n
        output : \n
        mark : int，-1表示违反约束，1表示不违反约束\n'''
        if len(x) != self.dim:
            raise ValueError('isOK：参数维度与测试函数维度不匹配')
        
        if x[0] < min[0] or x[0] > max[0] or x[1] < min[1] or x[1] > max[1] :
            raise ValueError('isOK: 参数已超出搜索空间')

        for g in self.constrain:
            if g(x)>0:
                return -1
        return 1

def plan1():

    f = TestFunction_G8()
    min = f.min
    max = f.max

    #遍历设计空间
    x, y = np.mgrid[min[0]:max[0]:100j, min[1]:max[1]:100j]
    s = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = [x[i, j], y[i, j]]
            s[i, j] = f.aim(a)



    #生成样本点
    sampleNum=20
    # lh=LatinHypercube(2,sampleNum,min,max)
    # realSample=lh.realSamples
    # np.savetxt('./Data/约束优化算法测试1/realSample.txt',realSample,delimiter=',')

    realSample = np.loadtxt('./Data/约束优化算法测试1/realSample.txt',delimiter=',')

    value=np.zeros(sampleNum)
    for i in range(0,sampleNum):
        a = [realSample[i, 0], realSample[i, 1]]
        value[i]=f.aim(a)
    
    #建立响应面
    kriging = Kriging()
    kriging.fit(realSample, value, min, max)
    
    print('正在优化theta参数....')
    theta = kriging.optimize(1000,'./Data/约束优化算法测试1/ADE_theta.txt')
    #计算预测值和方差
    preValue=np.zeros_like(x)
    varience=np.zeros_like(x)

    print('正在遍历响应面...')
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            a=[x[i, j], y[i, j]]
            preValue[i,j],varience[i,j]=kriging.transform(np.array(a))
    print('正在保存输出文件...')
    path = './Data/约束优化算法测试1/Kriging_Predicte_Model.txt'
    writeFile([x,y,preValue],[realSample,value],path)
    path = './Data/约束优化算法测试1/Kriging_Varience_Model.txt'
    writeFile([x,y,varience],[realSample,value],path)
    path = './Data/约束优化算法测试1/Kriging_True_Model.txt'
    writeFile([x,y,s],[realSample,value],path)    

    iterNum = 10    #加点数目
    for k in range(iterNum):
        print('第%d次加点'%(k+1))
        nextSample = kriging.nextPoint_Varience()
        realSample = np.vstack([realSample,nextSample])
        value = np.append(value,f.aim(nextSample))
        kriging.fit(realSample, value, min, max, theta)
        # kriging.optimize(100)

        #遍历响应面
        print('正在遍历响应面...')
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                a=[x[i, j], y[i, j]]
                preValue[i,j],varience[i,j]=kriging.transform(np.array(a))

        path = './Data/约束优化算法测试1/Kriging_Predicte_Model_%d.txt'%k
        writeFile([x,y,preValue],[realSample,value],path)
        path = './Data/约束优化算法测试1/Kriging_Varience_Model_%d.txt'%k
        writeFile([x,y,varience],[realSample,value],path)

    



if __name__=='__main__':
    plan1()
