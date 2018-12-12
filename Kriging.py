# ***************************************************************************
# Copyright (c) 2018 西安交通大学
# All rights reserved
# 
# 文件名称：Kriging.py
# 
# 摘    要：kriging模型预测算法
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
import ADE
import matplotlib.pyplot as plt
import DOE



class Kriging(object):
    '''
    初始化函数，构建对象，无参数。调用fit()函数训练模型。调用transform()返回预测值与方差。
    '''

    def fit(self,X,Y,min=None,max=None):
        """
        输入样本点，完成kriging建模,默认theta=[1...1],p=[2...2]\n
        输入：\n
        X : 样本点，n行d列，d是点的维度，n是点的数目\n
        Y : 样本点对应的值，n维向量\n    
        min&max : d维向量，表示拟合空间的范围，用以对数据进行归一化。如果min和max都等于none，说明点数据已经是归一化之后的数据\n

        输出：无，使用transform获取建模结果
        """        
        num=X.shape[0]
        d = X.shape[1]
        self.Y = Y.reshape((num, 1))

        if(min is None and max is None):
            self.X=X
        else:
            self.X=self.uniform(X,min,max)

        self.min = min
        self.max = max

        self.p = np.zeros(d)+2
        self.theta=np.zeros(d)+1
        self.R = np.zeros((num, num))

        self.log_likelihood(self.theta)


    def log_likelihood(self, theta):
        '''
        为了确定kriging模型中的超参数theta所设计的目标函数\n
        输入：\n
        theta : kriging超参数theta，用于确定设计空间各维度的光滑程度\n
        输出：\n
        对数似然函数的计算结果，值越大说明超参数theta更好
        '''
        points = self.X
        num = points.shape[0]
        d = points.shape[1]
        for i in range(d):
            self.theta[i]=theta[i]

        #计算相关矩阵
        for i in range(0, num):
            for j in range(0, i+1):
                if i==j:
                    self.R[i, j] = self.correlation(points[i, :], points[j, :])
                else:
                    self.R[i, j] = self.correlation(points[i, :], points[j, :])
                    self.R[j, i] = self.R[i, j]

        #计算相关矩阵的逆矩阵
        try:
            self.R_1 = np.linalg.inv(self.R)
        except np.linalg.linalg.LinAlgError as error:
            #相关矩阵不可逆，对应样本失败，返回负无穷
            return -np.inf
        F = np.zeros((num, 1)) + 1
        R_1 = self.R_1
        ys = self.Y

        beta0 = np.dot(F.T, R_1)
        denominator = np.dot(beta0, F)
        numerator = np.dot(beta0, ys)
        beta0 = numerator / denominator
        self.beta0 = beta0

        factor = ys - beta0 * F
        sigma2 = np.dot(factor.T, R_1)
        sigma2 = np.dot(sigma2, factor) / num
        self.sigma2 = sigma2
        det_R=np.abs(np.linalg.det(self.R))
        if(det_R==0 or sigma2==0):
            return -np.inf

        lgL=-num/2*np.log(sigma2**2)-0.5*np.log(det_R)
        # if(lgL>0):
        #     return -1000
        return lgL

    def optimize(self,maxGen,ADE_path=None):
        '''
        通过自适应差分进化算法，计算kriging模型中的theta参数，至于参数p设定为2。获取kriging的超参数后建立kriging的模型\n
        输入：\n
        maxGen : 差分进化计算的迭代次数\n
        ADE_path : 差分进化算法种群参数文件，如果为空使用拉丁超立方选择种群，否则从文本中读取。
        输出：\n
        无
        '''
        # 设置计算相关系数中所使用的theta
        d=self.X.shape[1]
        #min和max是差分进化算法的寻优空间，也就是theta的取值空间
        min=np.zeros(d)
        max=np.zeros(d)+1
        test = ADE.ADE(min, max, 100, 0.5, self.log_likelihood,False)
        if ADE_path is None:
            ind = test.evolution(maxGen=maxGen)
            import time
            timemark = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            path = './Data/ADE_Population_Argument_{0}.txt'.format(timemark)
            test.saveArg(path)
        else:
            ind = test.retrain(ADE_path,maxGen)
            test.saveArg(ADE_path)
        self.log_likelihood(ind.x)

    def transform(self,x):
        '''
        计算响应面中一点的期望和方差\n
        输入：\n
        x : d维向量，如果fit函数中的min和max非空，先归一化再计算\n
        输出：\n
        第一返回值为期望，第二返回值为方差
        '''
        num=self.X.shape[0]
        for i in range(x.shape[0]):
            x[i]=(x[i]-self.min[i])/(self.max[i]-self.min[i])
        r=np.zeros((num,1))
        for i in range(0,num):
            r[i]=self.correlation(self.X[i,:],x)
        F=np.zeros((num,1))+1
        R_1=self.R_1
        factor=self.Y-self.beta0*F
        y=np.dot(r.T,R_1)
        y=self.beta0+np.dot(y,factor)

        f1=np.dot(F.T,R_1)
        f1=(1-np.dot(f1,r))**2
        f2=np.dot(F.T,R_1)
        f2=np.dot(f2,F)
        f1=f1/f2
        f2=np.dot(r.T,R_1)
        f2=np.dot(f2,r)
        varience=self.sigma2*(1-f2+f1)

        return y, varience

    def uniform(self,points,min,max):
        """将样本点归一化
        输入：\n
        points : 样本点，n行d列，d是点的维度，n是点的数目\n
        min&max : d维向量，表示拟合空间的范围，用以对数据进行归一化。如果min和max都等于none，说明点数据已经是归一化之后的数据\n

        输出：\n
        归一化后的样本点
        """
        d=points.shape[1]
        num=points.shape[0]

        p=np.zeros(points.shape)
        for i in range(0,num):
            for j in range(0,d):
                p[i,j]=(points[i,j]-min[j])/(max[j]-min[j])
        return p

    def correlation(self,point1,point2):
        """获取两个样本点的相关系数\n
        输入：\n
        point1&point2 : 一维行向量,表示两个样本点\n
        输出：\n
        两个样本点的相关系数
        """
        d=point1.shape[0]
        R=np.zeros(d)
        theta=self.theta
        p=self.p
        for i in range(0,d):
            R[i]=-theta[i]*np.abs(point1[i]-point2[i])**p[i]
        return np.exp(np.sum(R))

def writeFile(graphData=[],pointData=[],path=None):
    '''
    将二维可展示数据写入文件\n
    input:\n
    graphData : 图像数据，list类型，包含三个矩阵————
        x : 矩阵，二维函数自变量x坐标值\n
        y : 矩阵，二维函数自变量y坐标值\n
        v : 矩阵，二维函数因变量v值\n
    pointData : 点数据，list类型，用以指示采样点的位置，包含两个变量————
        X : 矩阵，每一行代表一个样本点\n
        Y : 向量，存储与X行对应的样本点的采样值\n
    path : 文件路径，如果不指定按照当前时间设定文件名\n
    output:\n
    none
    '''
    if path is None:
        #获取当前时间作为文件名
        import time
        timemark = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        path = './Data/Kriging_Model_{0}.txt'.format(timemark)
    
    with open(path,'w') as file:
        num = len(graphData)
        if num==0:
            file.write('GraphData:%d\n'%num)
        else:
            file.write('GraphData:%d\n'%num)
            for i in range(num//3):
                x = graphData[i*3]
                y = graphData[i*3+1]
                v = graphData[i*3+2]

                row = x.shape[0]
                col = x.shape[1]
                file.write('row:%d\n'%row)
                file.write('col:%d\n'%col)        
                
                file.write('x:\n')
                for i in range(row):
                    for j in range(col):
                        file.write('%.18e,'%x[i,j])
                    file.write('\n') 
                
                file.write('y:\n')
                for i in range(row):
                    for j in range(col):
                        file.write('%.18e,'%y[i,j])
                    file.write('\n')     

                file.write('v:\n')
                for i in range(row):
                    for j in range(col):
                        file.write('%.18e,'%v[i,j])
                    file.write('\n')

        num = len(pointData)
        if num==0:
            file.write('PointData:%d\n'%num)
        else:
            file.write('PointData:%d\n'%num)
            for i in range(num//2):
                X = pointData[i*2]
                Y = pointData[i*2+1]
                row = X.shape[0]
                col = X.shape[1]
                file.write('row:%d\n'%row)
                file.write('col:%d\n'%col)        
                
                file.write('X:\n')
                for i in range(row):
                    for j in range(col):
                        file.write('%.18e,'%X[i,j])
                    file.write('\n') 
                    
                file.write('Y:\n')
                for i in range(row):
                    file.write('%.18e\n'%Y[i])
if __name__=="__main__":

    # leak函数
    # def func(X):
    #     x = X[0]
    #     y = X[1]
    #     return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
    #         -x ** 2 - y ** 2) - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
    # min = np.array([-3, -3])
    # max = np.array([3, 3])

    # Brain函数
    def func(x):
        pi=3.1415926
        y=x[1]-(5*x[0]**2)/(4*pi**2)+5*x[0]/pi-6
        y=y**2
        y+=10*(1-1/(8*pi))*np.cos(x[0])+10
        return y
    min = np.array([-5, 0])
    max = np.array([10, 15])

    sampleNum=21

    lh=DOE.LatinHypercube(2,sampleNum,min,max)
    sample=lh.samples
    realSample=lh.realSamples

    value=np.zeros(sampleNum)
    for i in range(0,sampleNum):
        a = [realSample[i, 0], realSample[i, 1]]
        value[i]=func(a)
    kriging = Kriging()
    kriging.fit(realSample, value, min, max)
    # kriging.optimize(100)

    prevalue=np.zeros_like(value)
    varience=np.zeros_like(value)
    for i in range(prevalue.shape[0]):
        a = [realSample[i, 0], realSample[i, 1]]
        prevalue[i],varience[i]=kriging.transform(np.array(a))
    
    print('实际值与预测值之差')
    print(np.abs(value-prevalue))

    x, y = np.mgrid[min[0]:max[0]:100j, min[1]:max[1]:100j]
    s = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = [x[i, j], y[i, j]]
            s[i, j] = func(a)
    
    preValue=np.zeros_like(x)
    varience=np.zeros_like(x)
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            a=[x[i, j], y[i, j]]
            preValue[i,j],varience[i,j]=kriging.transform(np.array(a))

    path = './Data/Kriging_True_Model.txt'
    writeFile([x,y,s],[realSample,value],path)
    path = './Data/Kriging_Predicte_Model.txt'
    writeFile([x,y,preValue],[realSample,value],path)
    path = './Data/Kriging_Varience_Model.txt'
    writeFile([x,y,varience],[realSample,value],path)