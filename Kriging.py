#!/usr/bin/env python
# -*- coding: utf-8 -*-
# # ***************************************************************************
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


import numpy as np
import ADE
import matplotlib.pyplot as plt
import DOE
from scipy.stats import norm
from scipy.special import comb


class Kriging(object):
    '''
    初始化函数，构建对象，无参数。调用fit()函数训练模型。调用transform()返回预测值与方差。
    '''


    def fit(self,X,Y,min=None,max=None,theta=None):
        """
        输入样本点，完成kriging建模,默认theta=[1...1],p=[2...2]\n
        输入：\n
        X : 样本点，n行d列，d是点的维度，n是点的数目\n
        Y : 样本点对应的值，n维向量\n    
        min&max : d维向量，表示拟合空间的范围，用以对数据进行归一化。如果min和max都等于none，说明点数据已经是归一化之后的数据\n
        theta : 一维向量，计算相关矩阵的参数，如果为空默认为[1,...,1]\n
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

        if theta is None:
            self.theta = np.zeros(d)+1
        else:
            self.theta = theta

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
            print('相关矩阵不可逆，对应theta设置错误')
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

    def optimize(self,maxGen,ADE_path):
        '''
        通过自适应差分进化算法，计算kriging模型中的theta参数，至于参数p设定为2。获取kriging的超参数后建立kriging的模型\n
        输入：\n
        maxGen : 差分进化计算的迭代次数\n
        ADE_path : 差分进化算法种群参数文件输出位置。\n
        输出：\n
        theta : theta的设定参数
        '''
        # 设置计算相关系数中所使用的theta
        d=self.X.shape[1]
        #min和max是差分进化算法的寻优空间，也就是theta的取值空间
        min=np.zeros(d)+0.001
        max=np.zeros(d)+100
        test = ADE.ADE(min, max, 100, 0.5, self.log_likelihood,False)
        ind = test.evolution(maxGen=maxGen)
        test.saveArg(ADE_path)

        self.log_likelihood(ind.x)

        return ind.x

    def transform(self,X):
        '''
        计算响应面中一点的期望和方差\n
        输入：\n
        x : d维向量，如果fit函数中的min和max非空，先归一化再计算\n
        输出：\n
        第一返回值为期望，第二返回值为方差的开方
        '''
        num=self.X.shape[0]
        x = np.zeros_like(X)
        if self.min is not None:
            for i in range(x.shape[0]):
                x[i]=(X[i]-self.min[i])/(self.max[i]-self.min[i])
        else:
            x = X

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
        if f2>1:
            f2 = 1
        s = self.sigma2*(1-f2+f1)
        if s<0:
            print(s)
        s=np.sqrt(s)    

        return y, s

    def get_Y(self,X):
        '''计算响应面设计点的期望\n
        input:\n
        x : 一维向量，采样点的坐标位置\n
        output:\n
        y : 响应面的估计值
        '''
        x = np.zeros_like(X)
        num=self.X.shape[0]
        if self.min is not None:
            for i in range(x.shape[0]):
                x[i]=(X[i]-self.min[i])/(self.max[i]-self.min[i])
        else:
            x = X
        r=np.zeros((num,1))
        for i in range(0,num):
            r[i]=self.correlation(self.X[i,:],x)
        F=np.zeros((num,1))+1
        R_1=self.R_1
        factor=self.Y-self.beta0*F
        y=np.dot(r.T,R_1)
        y=self.beta0+np.dot(y,factor)

        return y

    def get_S(self,X):
        '''计算响应面设计点的方差\n
        input:\n
        x : 一维向量，采样点的坐标位置\n
        output:\n
        s : 响应面的估计值
        '''
        num=self.X.shape[0]
        x = np.zeros_like(X)
        if self.min is not None:
            for i in range(x.shape[0]):
                x[i]=(X[i]-self.min[i])/(self.max[i]-self.min[i])
        else:
            x = X

        r=np.zeros((num,1))
        for i in range(0,num):
            r[i]=self.correlation(self.X[i,:],x)
        F=np.zeros((num,1))+1
        R_1=self.R_1

        f1=np.dot(F.T,R_1)
        f1=(1-np.dot(f1,r))**2
        f2=np.dot(F.T,R_1)
        f2=np.dot(f2,F)
        f1=f1/f2
        f2=np.dot(r.T,R_1)
        f2=np.dot(f2,r)
        if f2>1:
            f2 = 1
        s = self.sigma2*(1-f2+f1)
        if s<0:
            print(s)
        s=np.sqrt(s) 

        return s        

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

    def global_optimum(self,isMin=True):
        '''
        应用自适应差分进化算法，求解响应面的全局最优值\n
        input :\n
        isMin : 布尔变量\n
        output : \n
        不直接返回计算结果，将全局最优值存储在self.optimum，全局最优值的坐标存储在self.optimumLocation
        '''
        dim = self.X.shape[1]
        if self.min is None:
            min = np.zeros(dim)
            max = np.zeros(dim)+1
        else:
            min = self.min
            max = self.max
        ade = ADE.ADE(min, max, 100, 0.5, self.get_Y,isMin)
        print('搜索响应面最优值.....')
        opt_ind = ade.evolution(maxGen=100000)

        self.optimumLocation = opt_ind.x
        self.optimum = self.get_Y(opt_ind.x)

    def GEI(self,x):
        '''
        计算设计点x的GEI函数值，在调用该函数前必须调用global_optimum()获取全局最优值。\n
        input : \n
        x : 一维向量，采样点的坐标值\n
        output : \n
        gei : gei函数值
        '''
        g = self.g

        y,s = self.transform(x)
        u = (self.optimum-y)/s

        T = np.zeros(g+1)
        T[0] = norm.cdf(u)
        T[1] = -T[0]
        for k in range(2,g+1):
            T[k] = -u**(k-1)*norm.pdf(u)+(k-1)*T[k-2]

        gei = 0
        for k in range(g+1):
            gei += (-1)**k*comb(g,k)*u**(g-k)*T[k]
        gei = s**g*gei
        return gei 

    def EI(self,x):
        '''
        计算每一个点的EI函数，在调用该函数前必须调用global_optimum()获取全局最优值。\n
        input : \n
        x : 一维向量，采样点的坐标值\n
        output : \n
        ei : ei函数值
        '''
        xi = 0.01
        y,s = self.transform(x)
        if s != 0:
            u = (self.optimum-y-xi)/s
        else:
            u = 0
        
        ei = (self.optimum-y-xi)*norm.cdf(u)+s*norm.pdf(u)
        return ei

    def nextPoint_GEI(self,g):
        '''
        针对求解全局最小值的情况，应用GEI样本填充准则，寻找下一个采样点\n
        input:\n
        g : GEI函数的超参数，g越大倾向全局搜索，反之倾向于局部搜索。取值范围0~10的整数\n
        output:\n
        x : 下一个采样点的坐标位置
        
        '''
        self.g = g
        self.global_optimum()

        dim = self.X.shape[1]
        if self.min is None:
            min = np.zeros(dim)
            max = np.zeros(dim)+1
        else:
            min = self.min
            max = self.max
        ade = ADE.ADE(min, max, 100, 0.5, self.GEI,isMin=False)
        opt_ind = ade.evolution(maxGen=50000)
        # ade.saveArg('./Data/ADE_GEI_Optimum.txt')
        return opt_ind.x

    def nextPoint_Varience(self):
        '''选择方差最大的设计点作为下一轮的加点
        input : \n
        none \n
        output :\n
        x : 一维向量，代表需要计算的采样点'''
        dim = self.X.shape[1]
        if self.min is None:
            min = np.zeros(dim)
            max = np.zeros(dim)+1
        else:
            min = self.min
            max = self.max
        ade = ADE.ADE(min, max, 100, 0.5, self.get_S ,isMin=False)
        opt_ind = ade.evolution(maxGen=50000)
        # ade.saveArg('./Data/ADE_GEI_Optimum.txt')
        return opt_ind.x



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
                    for j in range(col-1):
                        file.write('%.18e,'%X[i,j])
                    file.write('%.18e\n'%X[i,col-1])

                    
                file.write('Y:\n')
                for i in range(row):
                    file.write('%.18e\n'%Y[i])


def func_leak(X):
    x = X[0]
    y = X[1]
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)

def func_Brain(x):
    pi=3.1415926
    y=x[1]-(5*x[0]**2)/(4*pi**2)+5*x[0]/pi-6
    y=y**2
    y+=10*(1-1/(8*pi))*np.cos(x[0])+10
    return y

#测试GEI加点函数
def test_Kriging_GEI():
    # 测试函数

    # leak函数
    # func = func_leak
    # min = np.array([-3, -3])
    # max = np.array([3, 3])

    # Brain函数
    func = func_Brain  
    min = np.array([-5, 0])
    max = np.array([10, 15])

    #遍历设计空间
    x, y = np.mgrid[min[0]:max[0]:100j, min[1]:max[1]:100j]
    s = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = [x[i, j], y[i, j]]
            s[i, j] = func(a)



    #生成样本点
    sampleNum=21
    # lh=DOE.LatinHypercube(2,sampleNum,min,max)
    # realSample=lh.realSamples
    # np.savetxt('./Data/realSample.txt',realSample,delimiter=',')
    realSample = np.loadtxt('./Data/realSample.txt',delimiter=',')

    value=np.zeros(sampleNum)
    for i in range(0,sampleNum):
        a = [realSample[i, 0], realSample[i, 1]]
        value[i]=func(a)
    
    #建立响应面
    kriging = Kriging()
    kriging.fit(realSample, value, min, max)
    # kriging.optimize(100)

    iterNum = 10    #加点数目
    preValue=np.zeros_like(x)
    varience=np.zeros_like(x)
    for k in range(iterNum):
        print('第%d次加点'%(k+1))
        nextSample = kriging.nextPoint_GEI(g=1)
        realSample = np.vstack([realSample,nextSample])
        value = np.append(value,func(nextSample))
        kriging = Kriging()
        kriging.fit(realSample, value, min, max)
        # kriging.optimize(100)

        #遍历响应面
        print('正在遍历响应面...')
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                a=[x[i, j], y[i, j]]
                preValue[i,j],varience[i,j]=kriging.transform(np.array(a))

        path = './Data/Kriging_Predicte_Model_%d.txt'%k
        writeFile([x,y,preValue],[realSample,value],path)
        path = './Data/Kriging_Varience_Model_%d.txt'%k
        writeFile([x,y,varience],[realSample,value],path)


    path = './Data/Kriging_True_Model.txt'
    writeFile([x,y,s],[realSample,value],path)

#测试kriging函数
def test_Kriging():
    # 测试函数

    # leak函数
    # func = func_leak
    # min = np.array([-3, -3])
    # max = np.array([3, 3])

    # Brain函数
    func = func_Brain  
    min = np.array([-5, 0])
    max = np.array([10, 15])

    #遍历设计空间
    x, y = np.mgrid[min[0]:max[0]:200j, min[1]:max[1]:200j]
    s = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = [x[i, j], y[i, j]]
            s[i, j] = func(a)



    #生成样本点
    sampleNum=21
    # lh=DOE.LatinHypercube(2,sampleNum,min,max)
    # realSample=lh.realSamples
    # np.savetxt('./Data/realSample.txt',realSample,delimiter=',')
    realSample = np.loadtxt('./Data/realSample1.txt',delimiter=',')
    sampleNum = realSample.shape[0]

    value=np.zeros(sampleNum)
    for i in range(0,sampleNum):
        a = [realSample[i, 0], realSample[i, 1]]
        value[i]=func(a)
    
    #建立响应面
    kriging = Kriging()
    kriging.fit(realSample, value, min, max)
    # kriging.optimize(100)


    #遍历响应面
    preValue=np.zeros_like(x)
    varience=np.zeros_like(x)    
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            a=[x[i, j], y[i, j]]
            preValue[i,j],varience[i,j]=kriging.transform(np.array(a))

    path = './Data/Kriging_Predicte_Model.txt'
    writeFile([x,y,preValue],[realSample,value],path)
    path = './Data/Kriging_Varience_Model.txt'
    writeFile([x,y,varience],[realSample,value],path)
    path = './Data/Kriging_True_Model.txt'
    writeFile([x,y,s],[realSample,value],path)    

def test_Kriging_GEI_Edition1():
    '''遍历GEI中g值由10到1的加点趋势变化'''

    #数据存储文件夹
    root_path = './Data/Kriging加点模型测试4'
    import os 
    if not os.path.exists(root_path):
        os.makedirs(root_path) 
    
    # Brain函数
    func = func_Brain  
    min = np.array([-5, 0])
    max = np.array([10, 15])

    #遍历设计空间
    x, y = np.mgrid[min[0]:max[0]:100j, min[1]:max[1]:100j]
    s = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = [x[i, j], y[i, j]]
            s[i, j] = func(a)

    path = root_path+'/Kriging_True_Model.txt'
    writeFile([x,y,s],[],path)

    #生成样本点
    for g in range(1,11):
        path = root_path+'/%d'%g
        if not os.path.exists(path):
            os.makedirs(path)
        realSample = np.loadtxt('./Data/realSample.txt',delimiter=',')
        sampleNum = realSample.shape[0]
        value=np.zeros(sampleNum)
        for i in range(0,sampleNum):
            a = [realSample[i, 0], realSample[i, 1]]
            value[i]=func(a)
        
        #建立响应面
        kriging = Kriging()
        kriging.fit(realSample, value, min, max)
        # kriging.optimize(100)

        iterNum = 20    #加点数目
        preValue=np.zeros_like(x)
        varience=np.zeros_like(x)
        for k in range(iterNum):
            print('\nGEI超参数g=%d,第%d次加点'%(g,(k+1)))
            nextSample = kriging.nextPoint_GEI(g)
            realSample = np.vstack([realSample,nextSample])
            value = np.append(value,func(nextSample))
            kriging = Kriging()
            kriging.fit(realSample, value, min, max)
            # kriging.optimize(100)

            #遍历响应面
            print('正在遍历响应面...')
            for i in range(0,x.shape[0]):
                for j in range(0,x.shape[1]):
                    a=[x[i, j], y[i, j]]
                    preValue[i,j],varience[i,j]=kriging.transform(np.array(a))

            path1 = path+'/Kriging_Predicte_Model_%d.txt'%k
            writeFile([x,y,preValue],[realSample,value],path1)
            path2 = path+'/Kriging_Varience_Model_%d.txt'%k
            writeFile([x,y,varience],[realSample,value],path2)



if __name__=="__main__":
    test_Kriging_GEI_Edition1()