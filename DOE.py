#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***************************************************************************
# Copyright (c) 2018 西安交通大学
# All rights reserved
# 
# 文件名称：DOE.py
# 
# 摘    要：实验设计方法，主要包含伪蒙特卡洛采样方法，准蒙特卡洛采样方法，拉丁超立方设计方法
#
# 创 建 者：上官栋栋
# 
# 创建日期：2018年11月21日
#
# 修改记录
# 日期  修改者   		版本     修改内容
# ------------- 		-------  ------------------------  
# ***************************************************************************



import numpy as np
import matplotlib.pyplot as plt


class PseudoMonteCarlo(object):
    """伪蒙特卡洛采样方法，考虑到均匀性，应用先分层再随机抽样的方法,采样空间在[0,1]**n之间"""
    def __init__(self,bins,num,min=None,max=None):
        """bins是n位向量，n是采样的维度，每一位的数据代表这一维分割的“箱子”数目；
        num代表这一个箱子中采样点数目，则采样点总数是bins[0]*bin[1]*...*bin[n-1]*num"""
        dimension=bins.shape[0]
        lengths=np.zeros(dimension)
        binNum=1
        for i in range(0,dimension):
            lengths[i]=1/bins[i]
            binNum=binNum*bins[i]
        binLocation=np.zeros((binNum,dimension))
        for i in range(0,binNum):
            index=self.transfer(bins,i)
            for j in range(0,dimension):
                binLocation[i,j]=index[j]*lengths[j]
        samples=np.zeros((binNum*num,dimension))
        for i in range(0,binNum):
            for j in range(0,num):
                random=np.random.uniform(0,1,dimension)
                for k in range(0,dimension):
                    samples[i*num+j,k]=binLocation[i,k]+lengths[k]*random[k]
        self.samples=samples
        if(min==None and max==None):
            self.realSamples=samples
        else:
            realSamples=np.zeros((samples.shape))
            for i in range(0,dimension):
                realSamples[:,i]=samples[:,i]*(max[i]-min[i])+min[i]
            self.realSamples=realSamples

    def transfer(self,weights,num):
        """根据weights给出的每一位的权重，将一个10进制数num转换为一个特殊进制的数
        如果num超过了weights所能表示的最大的数，则只返回可以表示的相应位置的数"""
        dimension=weights.shape[0]
        answer=np.zeros(dimension)
        remainder=num
        for i in range(0,dimension):
            answer[i]=remainder%weights[i]
            remainder=remainder//weights[i]
        return answer

class LatinHypercube(object):
    """拉丁超立方采样：在采样空间选取采样点，并减少采样点之间的相关性 \n
    输入：\n
    dimension:采样空间的维度\n
    num:样本的个数\n
    min&max:采样空间的上下限，默认样本空间是[0,1]**dimension\n
    
    输出：\n
    self.samples:标准化后在[0,1]内的采样点。二维矩阵，每一行是一个样本点\n
    self.realSamples:缩放到[min,max]内的实际采样点。二维矩阵，每一行是一个样本点"""    
    def __init__(self,dimension,num,min=None,max=None):
        """拉丁超立方采样：在采样空间选取采样点，并减少采样点之间的相关性 \n
        输入：\n
        dimension:采样空间的维度\n
        num:样本的个数\n
        min&max:采样空间的上下限，默认样本空间是[0,1]**dimension\n
        
        输出：\n
        self.samples:标准化后在[0,1]内的采样点。二维矩阵，每一行是一个样本点\n
        self.realSamples:缩放到[min,max]内的实际采样点。二维矩阵，每一行是一个样本点"""  
        if min is None:
            self.min = np.zeros(dimension)
        else:
            self.min = min
        
        if max is None:
            self.max = np.zeros(dimension)+1
        else:
            self.max = max  
        location=np.zeros((num,dimension))
        per=np.zeros((dimension*4,num))
        for i in range(0,per.shape[0]):
            per[i,:]=np.random.uniform(0,10,num)
        per=per.argsort(1)
        perNum=per.shape[0]
        isOK=True
        while(isOK):
            for i in range(0,dimension):
                rand=np.random.randint(0,perNum)
                location[:,i]=per[rand,:]
            try:
                R=np.cov(location.T)
                D=np.diag(np.diag(R)**-0.5)
                R=np.dot(D,np.dot(R,D))

                D=np.linalg.cholesky(R)
                G=np.dot(np.linalg.inv(D),location.T)
                isOK=False
            except np.linalg.linalg.LinAlgError as msg:
                print(msg)
                isOK=True
        list1=G.argsort(-1).T
        length=1/num
        sample=np.zeros((num,dimension))
        for i in range(0,num):
            random=np.random.uniform(0,1,dimension)
            sample[i,:]=random*length+list1[i,:]*length
        self.samples=sample

        if(min is None and max is None):
            self.realSamples=sample
        else:
            realSamples=np.zeros((sample.shape))
            for i in range(0,dimension):
                realSamples[:,i]=sample[:,i]*(max[i]-min[i])+min[i]
            self.realSamples=realSamples
    def writeFile(self,path):
        '''将samples写入path指定文件，注意存储的样本点是标准化后在[0,1]之内的。\n
        输入：\n
        path:文件目录，字符串\n
        输出：无'''
        np.savetxt(path,self.samples,delimiter=',')

    def show(self):
        '''展示self.realSamples的分布，只针对二维采样'''
        dimension = self.realSamples.shape[1]
        if dimension!=2:
            print("维度不等于2，不能展示")
        else:
            list_x = np.linspace(self.min[0],self.max[0],10+1)
            list_y = np.linspace(self.min[1],self.max[1],10+1)

            plt.scatter(self.realSamples[:,0],self.realSamples[:,1])
            plt.xlim((self.min[0],self.max[0]))
            plt.ylim((self.min[1],self.max[1]))
            plt.xticks(list_x)
            plt.yticks(list_y)
            plt.grid(True)
            plt.show()

if __name__=="__main__":
    # #伪蒙特卡洛采样方法测试
    # bin=np.array([10,10])
    # test=PseudoMonteCarlo(bin,1)
    # plt.scatter(test.samples[:,0],test.samples[:,1])
    # plt.show()
    #拉丁超立方采样方法测试
    test=LatinHypercube(dimension=2,num=10)
    test.show()