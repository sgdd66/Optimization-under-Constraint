# ***************************************************************************
# Copyright (c) 2018 西安交通大学
# All rights reserved
# 
# 文件名称：test.py
# 
# 摘    要：支持向量机
# 
# 创 建 者：上官栋栋
# 
# 创建日期：2018年11月9日
#
# 修改记录
# 日期  修改者   		版本     修改内容
# ------------- 		-------  ------------------------  
# ***************************************************************************

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SVM(object):

    def __init__(self,C,kernal=None):
        """
        初始化函数

        输入：\n
        kernal ： 核函数，如果不输入默认为欧式空间内积计算方法\n
        C ： 误分类惩罚因子

        输出：\n
        无返回
        """
        self.C=C
        if(kernal is None):
            self.Kernal=lambda x,y:np.sum(x*y)
        else:
            self.Kernal=kernal

    def fit(self,x,y):
        '''
        训练样本数据

        输入：\n
        x : 二维矩阵，一个样本占据一行\n
        y : 一维向量，只有-1和+1两种值，维度与x的行数相同，代表每个样本的分类结果\n

        输出：
        不返回
        '''

        #初始化变量
        row = x.shape[0]
        Gram = np.zeros((row, row))
        for i in range(row):
            for j in range(i, row):
                if i == j:
                    Gram[i, j] = self.Kernal(x[i, :], x[j, :])
                else:
                    Gram[i, j] = self.Kernal(x[i, :], x[j, :])
                    Gram[j, i] = Gram[i, j]
        self.Gram=Gram
        col=x.shape[1]
        self.w=np.zeros(col)
        self.b=0
        self.x=x
        self.y=y
        self.alpha = np.zeros(row)
        self.E=np.zeros(row)
        for i in range(row):
            self.E[i]=self.g(i)-self.y[i]

        #使用SMO算法求解
        self.SMO()

    def transform(self,x):
        '''
        根据训练模型预测样本类型

        输入变量：\n
        x : 一维向量，待预测的样本

        输出变量：\n
        +1 ：正例\n
        -1 ：负例
        '''
        y=self.Kernal(self.w,x)+self.b
        if y>0:
            return 1
        else:
            return -1

    def SMO(self):
        '''
        SMO求解算法，计算分割超平面的w和b

        输入：无

        输出：无
        '''
        #观测变量，存储每一轮迭代目标函数的值
        self.aim=[]
        #观测变量，存储每一迭代选择的变量索引
        self.variable=[]

        while(not self.canStop()):
            #更新两个变量的alpha
            i1,i2=self.getVariableIndex()
            self.variable.append([i1,i2])
            K11=self.Gram[i1,i1]
            K22=self.Gram[i2,i2]
            K12=self.Gram[i1,i2]
            eta=K11+K22-2*K12
            alpha1_old=self.alpha[i1]
            alpha2_old=self.alpha[i2]
            y1=self.y[i1]
            y2=self.y[i2]
            if(y1==y2):
                L=np.max([0,alpha2_old+alpha1_old-self.C])
                H=np.min([self.C,alpha2_old+alpha1_old])
            else:
                L=np.max([0,alpha2_old-alpha1_old])
                H=np.min([self.C,self.C+alpha2_old-alpha1_old])
            E1=self.E[i1]
            E2=self.E[i2]
            alpha2_new=alpha2_old+y2*(E1-E2)/eta
            if alpha2_new>H:
                alpha2_new=H
            elif alpha2_new<L:
                alpha2_new=L
            alpha1_new=alpha1_old+y1*y2*(alpha2_old-alpha2_new)
            self.alpha[i1]=alpha1_new
            self.alpha[i2]=alpha2_new
            #更新系数b
            b1_new=-E1-y1*K11*(alpha1_new-alpha1_old)-y2*K12*(alpha2_new-alpha2_old)+self.b
            b2_new=-E2-y1*K12*(alpha1_new-alpha1_old)-y2*K22*(alpha2_new-alpha2_old)+self.b
            if alpha1_new>0 and alpha1_new<self.C and alpha2_new>0 and alpha2_new<self.C:
                self.b = b1_new
            else:
                self.b=(b1_new+b2_new)/2

            #更新系数E
            for i in range(self.x.shape[0]):
                self.E[i]=self.g(i)-self.y[i]


            aim = 0
            for i in range(self.x.shape[0]):
                for j in range(self.x.shape[0]):
                    aim += self.alpha[i] * self.alpha[j] * self.y[i] * self.y[j] * self.Gram[i, j]
            aim -= np.sum(self.alpha)
            self.aim.append(aim)

            print(self.variable[-1],self.aim[-1])
        for i in range(self.x.shape[1]):
            self.w[i]=np.sum(self.alpha*self.y*self.x[:,i])

    def canStop(self):
        """
        SMO算法的终止条件
        
        输入：无

        输出：\n
        false : 不可终止\n
        true : 可终止\n
        """
        if np.sum(self.alpha*self.y)!=0:
            return False
        kkt=np.zeros((self.x.shape[0]))
        for i in range(self.x.shape[0]):
            kkt[i]=self.KKT(i)
        if np.sum(np.abs(kkt))>0.00001:
            return False
        return True

    def g(self,i):
        '''
        样本点预测函数

        输入：\n
        i : 样本点在样本集中的编号，即第i行样本

        输出：\n
        样本点预测值
        '''
        return np.sum(self.alpha*self.y*self.Gram[:,i])+self.b

    def getVariableIndex(self):
        """
        选择两个变量，第一个变量是违反kkt条件最严重的项，第二个是可以使第一项获得最大提升的项
        
        输入：无
        
        输出：\n
        index1 : 第一个变量的索引号\n
        index2 : 第二个变量的索引号
        """

        row=self.x.shape[0]
        kkt=np.zeros(row)
        for i in range(row):
            kkt[i]=self.KKT(i)
        kktSorted=np.sort(kkt)    
        E_sorted_list=np.sort(self.E)        

           
        index1=np.where(kkt==kktSorted[row-1])[0][0]
        e1=self.E[index1]
        
        if e1>0:
            index2=np.where(E_sorted_list[0]==self.E)[0][0]
            if index2 == index1:
                #如果最小值有多个相同的值，则应该使用这种方法
                if len(np.where(E_sorted_list[0]==self.E)[0])>1:
                    index2 = np.where(E_sorted_list[0]==self.E)[0][1]
                else:
                    index2=np.where(E_sorted_list[1]==self.E)[0][0]
        else:
            index2=np.where(E_sorted_list[row-1]==self.E)[0][0]
            if index2 == index1:
                if len(np.where(E_sorted_list[row-1]==self.E)[0])>1:
                    index2 = np.where(E_sorted_list[row-1]==self.E)[0][1] 
                else:
                    index2=np.where(E_sorted_list[row-2]==self.E)[0][0] 

        #如果选择的优化变量与之前两轮的优化变量相同，将会重新选择
        if len(self.variable)>2:
            last_index1 = self.variable[-1]
            last_index2 = self.variable[-2]
            if index1 == last_index1[0] and index2== last_index1[1] and \
                index1 == last_index2[0] and index2 == last_index2[1]:

                if len(np.where(kkt==kktSorted[row-1])[0])>1:
                    index1 = np.where(kkt==kktSorted[row-1])[0][1]
                else:
                    index1=np.where(kkt==kktSorted[row-2])[0][0]
                e1=self.E[index1]
                
                if e1>0:
                    index2=np.where(E_sorted_list[0]==self.E)[0][0]
                    if index2 == index1:
                        #如果最小值有多个相同的值，则应该使用这种方法
                        if len(np.where(E_sorted_list[0]==self.E)[0])>1:
                            index2 = np.where(E_sorted_list[0]==self.E)[0][1]
                        else:
                            index2=np.where(E_sorted_list[1]==self.E)[0][0]
                else:
                    index2=np.where(E_sorted_list[row-1]==self.E)[0][0]
                    if index2 == index1:
                        if len(np.where(E_sorted_list[row-1]==self.E)[0])>1:
                            index2 = np.where(E_sorted_list[row-1]==self.E)[0][1] 
                        else:
                            index2=np.where(E_sorted_list[row-2]==self.E)[0][0]              
                


        return index1,index2

    def KKT(self,i):
        '''
        检验变量是否满足KKT条件，若不满足返回与标准值（1）之差的绝对值，否则返回0
        
        输入：\n
        i : 样本编号

        输出：
        int类型，0代表满足KKT条件，其他值代表与标准值的偏差
        '''
        alpha=self.alpha[i]
        y=self.y[i]
        g=self.g(i)
        if(alpha==0):
            if(y*g>=1):
                return 0
            else:
                return np.abs(1-y*g)
        elif(alpha==self.C):
            if(y*g<=1):
                return 0
            else:
                return np.abs(y*g-1)
        else:
            if(y*g==1):
                return 0
            else:
                return np.abs(y*g-1)

def test_Sample1():
    '''测试样本1：正例与反例线性可分'''
    x1 = 3
    y1 = 5

    x2 = 5
    y2 = 3

    R1 = 1
    R2 = 1

    R1_list = np.random.uniform(0,1,50)*R1
    R2_list = np.random.uniform(0,1,50)*R2

    angle1_list = np.random.uniform(0,1,50)*2*np.pi
    angle2_list = np.random.uniform(0,1,50)*2*np.pi

    x1_list = R1_list*np.cos(angle1_list)+x1
    y1_list = R1_list*np.sin(angle1_list)+y1

    x2_list = R2_list*np.cos(angle2_list)+x2
    y2_list = R2_list*np.sin(angle2_list)+y2 

    # fig = plt.figure(0)
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(x1_list,y1_list)
    # ax1.scatter(x2_list,y2_list)
    # plt.show()

    x1_list = x1_list.reshape((-1,1))
    y1_list = y1_list.reshape((-1,1))
    sample_positive = np.hstack((x1_list,y1_list,np.zeros((50,1))+1))

    x2_list = x2_list.reshape((-1,1))
    y2_list = y2_list.reshape((-1,1))
    sample_negative = np.hstack((x2_list,y2_list,np.zeros((50,1))-1))

    sample = np.vstack((sample_negative,sample_positive))
    np.savetxt('./Data/sample1.csv',sample,delimiter=',')

    


if __name__=="__main__":
    # test_Sample1()

    x=np.array([[1,2],[2,3],[3,3],[2,1],[3,2]])

    y=np.array([1,1,1,-1,-1])

    svm=SVM(1)
    svm.fit(x,y)
    plt.scatter(x[:,0],x[:,1])
    x=np.linspace(0,4,100)
    y=(svm.w[0]*x+svm.b)/-svm.w[1]
    plt.plot(x,y)
    plt.show()
