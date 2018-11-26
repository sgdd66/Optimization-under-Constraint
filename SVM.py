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
    """
    初始化函数

    输入：\n
    kernal ： 核函数，如果不输入默认为欧式空间内积计算方法\n
    C ： 误分类惩罚因子

    输出：\n
    无返回
    """
    def __init__(self,C,kernal=None):

        self.C=C
        if(kernal is None):
            self.Kernal=lambda x,y:np.sum(x*y)
        else:
            self.Kernal=kernal

    def fit(self,x,y,maxIter=100):
        '''
        训练样本数据

        输入：\n
        x : 二维矩阵，一个样本占据一行\n
        y : 一维向量，只有-1和+1两种值，维度与x的行数相同，代表每个样本的分类结果\n

        输出：
        不返回
        '''

        #初始化变量
        self.sampleSum = x.shape[0]
        Gram = np.zeros((self.sampleSum, self.sampleSum))
        for i in range(self.sampleSum):
            for j in range(i, self.sampleSum):
                if i == j:
                    Gram[i, j] = self.Kernal(x[i, :], x[j, :])
                else:
                    Gram[i, j] = self.Kernal(x[i, :], x[j, :])
                    Gram[j, i] = Gram[i, j]
        self.Gram=Gram
        self.sampleDim=x.shape[1]
        self.b=0
        self.x=x
        self.y=y
        self.alpha = np.zeros(self.sampleSum)
        self.E=np.zeros(self.sampleSum)
        self.kkt=np.zeros((self.sampleSum))
        for i in range(self.sampleSum):
            self.E[i]=self.g(i)-self.y[i]
        
        self.maxIter = maxIter
        #观测变量，存储每一轮迭代目标函数的值
        self.aim=[]
        #观测变量，存储每一迭代选择的变量索引
        self.variable=[]
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

        kernal = np.zeros(self.sampleSum)
        for i in range(self.sampleSum):
            kernal[i]=self.Kernal(self.x[i,:],x)
        y=np.sum(self.alpha*self.y*kernal)+self.b
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

        while(not self.canStop()):

            i1,i2=self.getVariableIndex()
            self.refresh(i1,i2)

            #如果目标值没有足够下降，更换index2
            if len(self.aim)<2:
                continue
            else:
                #支持向量索引
                sv_index = np.where(self.alpha!=0)[0]      
                #非支持向量索引 
                nsv_index = np.where(self.alpha==0)[0]
                j1 = 0

            while(self.aim[-2]-self.aim[-1]<10**-5):

                #首先遍历支持向量，然后遍历非支持向量
                if j1 < len(sv_index):
                    if sv_index[j1]==i1:
                        j1+=1
                        continue
                    else:
                        i2 = sv_index[j1]
                        self.refresh(i1,i2)
                        j1+=1
                elif j1 < self.sampleSum:
                    j2 = j1-len(sv_index)
                    if nsv_index[j2]==i1:
                        j1+=1
                        continue
                    else:
                        i2 = nsv_index[j2]
                        self.refresh(i1,i2)
                        j1+=1
                else:
                    print("所有变量均不能使第一变量{0}充分下降，更换变量".format(i1))
                    break

        self.saveArg()

    def saveArg(self):
        '''存储计算需要的内部参数，方便训练与模型恢复'''
        #获取当前时间作为文件名
        import time
        timemark = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        path = './Data/SVM_Argument_{0}.txt'.format(timemark)
        with open(path,'w') as file:
            file.write('C:%.18e\n'%self.C)
            file.write('sample sum:{0}\n'.format(self.sampleSum))
            file.write('sample dimension:{0}\n'.format(self.sampleDim))

            file.write('x:\n')
            for i in range(self.sampleSum):
                for j in range(self.sampleDim):
                    file.write('%.18e,'%self.x[i,j])
                file.write('\n') 

            file.write('y:\n')
            for j in range(self.sampleSum):
                file.write("%d\n"%self.y[j])

            file.write('Gram matrix:\n')
            for i in range(self.sampleSum):
                for j in range(self.sampleSum):
                    file.write('%.18e,'%(self.Gram[i,j]))
                file.write('\n')

            file.write('b:%.18e\n'%(self.b))

            file.write('alpha:\n')
            for i in range(self.sampleSum):
                file.write('%d : %.18e\n'%(i,self.alpha[i]))


            file.write('E:\n')
            for i in range(self.sampleSum):
                file.write('%d : %.18e\n'%(i,self.E[i]))


            file.write('kkt:\n')
            for i in range(self.sampleSum):
                file.write('%d : %.18e\n'%(i,self.kkt[i]))
      

            file.write('aim:%d\n'%len(self.aim))
            for i in range(len(self.aim)):
                file.write('%.18e\n'%self.aim[i])

            file.write('variable:%d\n'%len(self.variable))
            for i in range(len(self.variable)):
                file.write('%d,%d\n'%(self.variable[i][0],self.variable[i][1]))




        # #初始化变量

        # self.sampleSum = x.shape[0]
        # self.sampleDim=x.shape[1]

        # self.x=x
        # self.y=y
        # self.Gram=Gram
        
        # self.b=0

        # self.alpha = np.zeros(self.sampleSum)
        # self.E=np.zeros(self.sampleSum)
        # self.kkt=np.zeros((self.sampleSum))
 

        # #观测变量，存储每一轮迭代目标函数的值
        # self.aim=[]
        # #观测变量，存储每一迭代选择的变量索引
        # self.variable=[]

    def retrain(self,path,maxIter=100):
        '''读取参数文件重新训练\n

        输入：\n
        path : 参数文件的文件路径\n
        maxIter : 新一轮训练的最大迭代次数\n

        输出：无\n
        '''
        import re 
        with open(path,'r') as file:
            texts = file.readlines()


            reg_int = re.compile(r'-?\d+')
            reg_float = re.compile(r'-?\d\.\d+e[\+,-]\d+')      

            self.C = float(reg_float.search(texts[0]).group(0))
            self.sampleSum = int(reg_int.search(texts[1]).group(0))
            self.sampleDim = int(reg_int.search(texts[2]).group(0))

            pos = 4
            self.x = np.zeros((self.sampleSum,self.sampleDim))
            for i in range(self.sampleSum):
                text_list = reg_float.findall(texts[pos+i])
                for j in range(self.sampleDim):
                    self.x[i,j] = float(text_list[j])
            
            pos = pos+self.sampleSum+1
            self.y = np.zeros(self.sampleSum)
            for i in range(self.sampleSum):
                self.y[i] = int(reg_int.search(texts[pos+i]).group(0))
                          
            pos = pos+self.sampleSum+1
            self.Gram = np.zeros((self.sampleSum,self.sampleSum))
            for i in range(self.sampleSum):
                text_list = reg_float.findall(texts[pos+i])
                for j in range(self.sampleSum):
                    self.Gram[i,j] = float(text_list[j])    

            pos = pos+self.sampleSum
            self.b = float(reg_float.search(texts[pos]).group(0))        

            pos += 2
            self.alpha = np.zeros(self.sampleSum)
            for i in range(self.sampleSum):
                self.alpha[i] = float(reg_float.search(texts[pos+i]).group(0))
           
            pos += self.sampleSum+1
            self.E = np.zeros(self.sampleSum)
            for i in range(self.sampleSum):
                self.E[i] = float(reg_float.search(texts[pos+i]).group(0))

            pos += self.sampleSum+1
            self.kkt = np.zeros(self.sampleSum)
            for i in range(self.sampleSum):
                self.kkt[i] = float(reg_float.search(texts[pos+i]).group(0))

        self.variable = []
        self.aim = []
        self.maxIter = maxIter
        self.SMO()

    def refresh(self,i1,i2):
        '''根据i1，i2更新所有参数'''
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
        if alpha1_new>0 and alpha1_new<self.C:
            self.b = b1_new
        elif alpha2_new>0 and alpha2_new<self.C:
            self.b = b2_new
        else:
            self.b=(b1_new+b2_new)/2

        #更新系数E
        for i in range(self.sampleSum):
            self.E[i]=self.g(i)-self.y[i]


        aim = 0
        for i in range(self.sampleSum):
            for j in range(self.sampleSum):
                aim += self.alpha[i] * self.alpha[j] * self.y[i] * self.y[j] * self.Gram[i, j]
        aim /= 2.0
        aim -= np.sum(self.alpha)
        self.aim.append(aim)

        print([i1,i2],self.aim[-1])

    def canStop(self):
        """
        SMO算法的终止条件
        
        输入：无

        输出：\n
        false : 不可终止\n
        true : 可终止\n
        """
        
        if len(self.aim)>=self.maxIter:
            print("因达到最大迭代次数，计算终止。")
            return True

        for i in range(self.sampleSum):
            self.kkt[i]=self.KKT(i)
        if np.sum(np.abs(self.kkt))>0.001:
            return False
        else:
            print("满足kkt条件，达到最优值。当前迭代次数：{0}".format(len(self.aim)))
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


        kkt=self.kkt
        kktSorted=np.sort(kkt)    

        #根据kktSorted求得对应的index序列
        index_list = []
        while len(index_list)<self.sampleSum:
            num = len(index_list)
            index_list.extend(np.where(kkt==kktSorted[num])[0])   

        #从支持向量中选择kkt违反条件最严重的
        index1 = np.nan
        for i in range(len(index_list)-1,-1,-1):
            index = index_list[i]
            if self.alpha[index]>0 and self.alpha[index]<self.C and kkt[index]>10**-5:
                index1 = index
                break
        if index1 is np.nan:
            index1 = index_list[-1]
        
        #如果目标函数没有充分下降，而且index1和上一轮相同，更换index1
        if len(self.aim)>5 and self.aim[-5]-self.aim[-1]<10**-5 and index1==self.variable[-1][0]:
            index_list.remove(index1)
            index1 = np.nan
            for i in range(len(index_list)-1,-1,-1):
                index = index_list[i]
                if self.alpha[index]>0 and self.alpha[index]<self.C and kkt[index]>10**-5:
                    index1 = index
                    break
            if index1 is np.nan:
                index1 = index_list[-1]

        e1=self.E[index1]

        max_error_diff = 0
        for i in range(self.sampleSum):
            if i == index1:
                continue
            if np.abs(self.E[i]-e1)>max_error_diff:
                index2 = i


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

    def show(self):
        '''
        展示SMO算法求解的结果，仅限于二维展示
        '''
        calc_outcome = self.E+self.y
        mark_outcome = self.y

        #正例支持向量
        support_vector_positive = []
        #反例支持向量
        support_vector_negative = []

        #正例判对
        pos_true = []
        #正例判错
        pos_false = []
        #反例判对
        neg_true = []
        #反例判错
        neg_false = []

        #落在分割超平面上的点
        mid = []


        for i in range(self.sampleSum):
            #支持向量
            if self.alpha[i]>0 and self.alpha[i]<self.C:
                if calc_outcome[i]>0:
                    support_vector_positive.append(self.x[i,:])
                else:
                    support_vector_negative.append(self.x[i,:])
            else:
                #非支持向量
                if calc_outcome[i]==0:
                    mid.append(self.x[i,:])
                elif mark_outcome[i]>0 and calc_outcome[i]>0:
                    pos_true.append(self.x[i,:])
                elif mark_outcome[i]>0 and calc_outcome[i]<0:
                    pos_false.append(self.x[i,:])
                elif mark_outcome[i]<0 and calc_outcome[i]>0:
                    neg_false.append(self.x[i,:])
                elif mark_outcome[i]<0 and calc_outcome[i]<0:
                    neg_true.append(self.x[i,:])

        if len(support_vector_negative)>0:
            support_vector_negative = np.array(support_vector_negative)
            plt.scatter(support_vector_negative[:,0],support_vector_negative[:,1],c='b',marker='v')
        if len(support_vector_positive)>0:
            support_vector_positive = np.array(support_vector_positive)
            plt.scatter(support_vector_positive[:,0],support_vector_positive[:,1],c='r',marker='v')
        if len(mid)>0:
            mid = np.array(mid)
            plt.scatter(mid[:,0],mid[:,1],c='k',marker='o')
        if len(pos_false)>0:
            pos_false = np.array(pos_false)
            plt.scatter(pos_false[:,0],pos_false[:,1],c='r',marker='x')        
        if len(pos_true)>0:
            pos_true = np.array(pos_true)
            plt.scatter(pos_true[:,0],pos_true[:,1],c='r',marker='.')
        if len(neg_false)>0:
            neg_false = np.array(neg_false)
            plt.scatter(neg_false[:,0],neg_false[:,1],c='b',marker='x')
        if len(neg_true)>0:
            neg_true = np.array(neg_true)
            plt.scatter(neg_true[:,0],neg_true[:,1],c='b',marker='.')            

        # x_min = min(self.x[:,0])
        # x_max = max(self.x[:,1])
        # x = np.linspace(x_min,x_max,100)
        # w = np.zeros(2)
        # for i in range(self.sampleDim):
        #     w[i] = np.sum(self.alpha*self.y*self.x[:,i])
        # y = (w[0]*x+svm.b)/-w[1]
        # plt.plot(x,y)

        import time
        timemark = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        path = './Data/SVM_Photo_{0}.png'.format(timemark)
        plt.savefig(path)
        plt.show()

def test_Sample0():
    x=np.array([[1,2],[2,3],[3,3],[2,1],[3,2]])
    y=np.array([1,1,1,-1,-1])
    y=y.reshape((-1,1))
    sample = np.hstack((x,y))
    np.savetxt('./Data/sample0.csv',sample,delimiter=',')

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

def test_Sample2():
    '''测试样本2，正例与反例线性不可分'''
    x1 = 3
    y1 = 3

    x2 = 4
    y2 = 4

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

    fig = plt.figure(0)
    ax1 = fig.add_subplot(111)
    ax1.scatter(x1_list,y1_list)
    ax1.scatter(x2_list,y2_list)
    plt.show()

    x1_list = x1_list.reshape((-1,1))
    y1_list = y1_list.reshape((-1,1))
    sample_positive = np.hstack((x1_list,y1_list,np.zeros((50,1))+1))

    x2_list = x2_list.reshape((-1,1))
    y2_list = y2_list.reshape((-1,1))
    sample_negative = np.hstack((x2_list,y2_list,np.zeros((50,1))-1))

    sample = np.vstack((sample_negative,sample_positive))
    np.savetxt('./Data/sample2.csv',sample,delimiter=',')  
    
def test_Sample3():
    '''测试样本3，正例与反例非线性可分'''
    import DOE
    min = [0,0]
    max = [2,2]
    latinSamples = DOE.LatinHypercube(dimension=2,num=100,min=min,max=max)
    samples = latinSamples.realSamples
    mark = np.zeros(100)
    f = lambda x:-x**2+2*x+0.3
    for index,sample in enumerate(samples):
        if sample[1]>f(sample[0]):
            mark[index]=1
        else:
            mark[index]=-1

    sample_pos = samples[np.where(mark==1)[0],:]
    sample_neg = samples[np.where(mark==-1)[0],:]
    plt.scatter(sample_pos[:,0],sample_pos[:,1])
    plt.scatter(sample_neg[:,0],sample_neg[:,1])
    x = np.linspace(min[0],max[0],100)
    y = f(x)  
    plt.plot(x,y)
    plt.show()

    mark = mark.reshape((-1,1))
    samples = np.hstack((samples,mark))
    np.savetxt('./Data/sample3.csv',samples,delimiter=',')




if __name__=="__main__":
    # test_Sample3()

    f=lambda x,y:(np.dot(x,y)+1)**3

    # path = './Data/sample3.csv'
    # data = np.loadtxt(path,delimiter=',')
    # x = data[:,0:2]
    # y = data[:,2]
    # svm=SVM(1,kernal=f)
    # svm.fit(x,y,maxIter=1000)
    # svm.show()

    svm = SVM(1,kernal=f)
    path = '/home/sgdd/Optimization-under-Constraint/Data/SVM_Argument_2018-11-26_17-17-04.txt'
    svm.retrain(path,maxIter=5000)
    svm.show()

