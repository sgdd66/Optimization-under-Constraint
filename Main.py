#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


from Kriging import Kriging,writeFile,filterSamples
from SVM import SVM,Kernal_Gaussian,Kernal_Polynomial
from DOE import LatinHypercube
import numpy as np
from ADE import ADE
from test import test28

class TestFunction_G8(object):
    '''测试函数G8:\n
    变量维度 : 2\n
    搜寻空间 : 0.001 ≤ xi ≤ 10, i = 1, 2.\n
    全局最小值 : x∗ = (1.2279713, 4.2453733) , f (x∗) = -0.095825.'''

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
        
        if x[0] < self.min[0] or x[0] > self.max[0] or x[1] < self.min[1] or x[1] > self.max[1] :
            raise ValueError('isOK: 参数已超出搜索空间')

        for g in self.constrain:
            if g(x)>0:
                return -1
        return 1

class TestFunction_G4(object):
    '''
    测试函数G4 \n
    变量维度 : 5\n
    搜索空间 : l=(78,33,27,27,27),u=(102,45,45,45,45),li<xi<ui,i=1...5\n
    全局最优值 : x* =  (78,33,29.996,45,36.7758),f(x*) = -30665.539
    '''
    def __init__(self):
        '''建立目标和约束函数'''
        self.aim = lambda x:5.3578547*x[2]**2+0.8356891*x[0]*x[4]+37.293239*x[0]-40792.141
        self.u = lambda x:85.334407+0.0056858*x[1]*x[4]+0.0006262*x[0]*x[3]-0.0022053*x[2]*x[4]
        self.v = lambda x:80.51249+0.0071317*x[1]*x[4]+0.0029955*x[0]*x[1]+0.0021813*x[2]**2
        self.w = lambda x:9.300961+0.0047026*x[2]*x[4]+0.0012547*x[0]*x[2]+0.0019085*x[2]*x[3]

        #约束函数，小于等于0为满足约束
        g1 = lambda x : self.u(x)-92
        g2 = lambda x : -self.u(x)
        g3 = lambda x : self.v(x)-110
        g4 = lambda x : -self.v(x)+90
        g5 = lambda x : self.w(x)-25
        g6 = lambda x : -self.w(x)+20
        self.constrain = [g1,g2,g3,g4,g5,g6]

        self.dim = 5
        self.min = [78,33,27,27,27]
        self.max = [102,45,45,45,45]

        self.optimum = [78,33,29.995,45,36.7758]

    def isOK(self,x):
        '''检查样本点x是否违背约束，是返回-1，否返回1\n
        input : \n
        x : 样本点，一维向量\n
        output : \n
        mark : int，-1表示违反约束，1表示不违反约束\n'''
        if len(x) != self.dim:
            raise ValueError('isOK：参数维度与测试函数维度不匹配')
        
        for i in range(self.dim):
            if x[i] < self.min[i] or x[i] > self.max[i]:
                raise ValueError('isOK: 参数已超出搜索空间')

        for g in self.constrain:
            if g(x)>0:
                return -1
        return 1

def stepA_1():
    '''
    步骤A的第一种版本，加点过程是确定数目的，并以方差为优化目标，降低全局的不确定性
    '''
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

    #检测样本点中是否有可行解，如果没有继续加点
    mark = np.zeros(realSample.shape[0])
    for i in range(realSample.shape[0]):
        mark[i] = f.isOK(realSample[i,:])
    
    if np.sum(mark==1)>0:
        value = value.reshape((-1,1))
        mark = mark.reshape((-1,1))
        storeData = np.hstack((realSample,value,mark))
        np.savetxt('./Data/约束优化算法测试1/samples1.txt',storeData)
        return

    
    i = 0
    while mark[-1]==-1:
        i += 1
        print('第%d次加点'%(iterNum+i))
        nextSample = kriging.nextPoint_Varience()
        realSample = np.vstack([realSample,nextSample])
        value = np.append(value,f.aim(nextSample))
        mark = np.append(mark,f.isOK(nextSample))
        kriging.fit(realSample, value, min, max, theta)
        # kriging.optimize(100)

        #遍历响应面
        print('正在遍历响应面...')
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                a=[x[i, j], y[i, j]]
                preValue[i,j],varience[i,j]=kriging.transform(np.array(a))

        path = './Data/约束优化算法测试1/Kriging_Predicte_Model_%d.txt'%(iterNum+i)
        writeFile([x,y,preValue],[realSample,value],path)
        path = './Data/约束优化算法测试1/Kriging_Varience_Model_%d.txt'%(iterNum+i)
        writeFile([x,y,varience],[realSample,value],path)

    value = value.reshape((-1,1))
    mark = mark.reshape((-1,1))
    storeData = np.hstack((realSample,value,mark))
    np.savetxt('./Data/约束优化算法测试1/samples1.txt',storeData)

def stepB_1():
    '''
    步骤B的第一种版本，固定加点次数

    '''  
    '''测试svm的加点算法，固定加点次数'''

    #理论分割函数
    f = TestFunction_G8()
    data = np.loadtxt('./Data/约束优化算法测试1/samples1.txt')
    samples = data[:,0:2]
    mark = data[:,3]

    #多项式核函数指数为5
    svm=SVM(5,kernal=Kernal_Polynomial,path = './Data/约束优化算法测试1')
    svm.fit(samples,mark,maxIter=50000)
    svm.show()

    T1_list = [1,0.8,0.6,0.4]
    pointNum = np.zeros(len(T1_list)+1)
    pointNum[0] = samples.shape[0]

    for k in range(len(T1_list)):
        if k < 4:
            new_x = svm.infillSample1(0.5,T1_list[k],f.min,f.max,[40,40])
        else:
            new_x = svm.infillSample2(0.5,T1_list[k],f.min,f.max,[40,40])            
        num = new_x.shape[0]
        if num == 0:
            print('当T1设置为%.2f时，加点数目为0'%T1_list[k])
            pointNum[k+1] = samples.shape[0]
            continue
        new_y = np.zeros(num)
        for i in range(num):
            new_y[i] = f.isOK(new_x[i,:])
        samples = np.vstack((samples,new_x))
        mark = np.append(mark,new_y)
        svm.fit(samples,mark,100000)
        svm.show()
        pointNum[k+1] = samples.shape[0]

    print('样本点数目：')
    print(pointNum)
    print('加点结束')    

def FeasibleSpace(svm, labelNum,edgeRatio):
    '''
    提取包含支持向量机的可行域（正例）的超立方空间。\n
    input : \n
    svm : SVM类实例\n
    labelNum : 每个维度划分的分割点数目\n
    edgeRatio : 取值0~1的浮点数，表示一个维度上，可行域向外围延展的区域与可行域在该维度投影区域的比例\n
    output : \n
    space_min,space_max,第一项是超立方空间的下限，第二项是超立方空间的上限\n'''
    dim = svm.x.shape[1]

    space_min = np.zeros(dim)
    space_max = np.zeros(dim)
    for i in range(dim):
        space_min[i] = min(svm.x[:,i])
        space_max[i] = max(svm.x[:,i])

    pointNum = 1
    for i in range(dim):
        pointNum *= labelNum[i]
    coordinate = np.zeros((pointNum,dim))
    mark = np.zeros(pointNum)    
    for i in range(dim):
        coordinate[:,i]+=space_min[i]

    for n in range(pointNum):
        index = n
        for m in range(dim):
           i = index%labelNum[m]
           coordinate[n,m]+=(space_max[m]-space_min[m])/(labelNum[m]-1)*i
           index =  index//labelNum[m]
           if index == 0:
               break
        mark[n] = svm.transform(coordinate[n,:])

    posIndex = np.where(mark>0)[0]
    posCoordinate = coordinate[posIndex,:]

    space_min = np.min(posCoordinate,0)
    space_max = np.max(posCoordinate,0)

    space_length = space_max - space_min
    space_min = space_min - space_length*edgeRatio
    space_max = space_max + space_length*edgeRatio

    return space_min,space_max

def stepC_1():
    '''步骤C的第一种版本'''
    #违反约束的惩罚系数
    penalty = 10000
    root_path = './Data/约束优化算法测试1/stepC_5'
    import os 
    if not os.path.exists(root_path):
        os.makedirs(root_path) 


    # 加载支持向量机
    svm=SVM(5,kernal=Kernal_Polynomial,path =root_path )
    svm.retrain(root_path+'/SVM_Argument_2019-01-15_11-41-00.txt',maxIter=1)
    
    # 提取已采样样本的坐标，值，是否违反约束的标志
    allSamples = svm.x
    allValue = np.zeros(allSamples.shape[0])
    allMark = np.zeros(allSamples.shape[0])
    testFunc = TestFunction_G8()
    for i in range(allSamples.shape[0]):
        allValue[i] = testFunc.aim(allSamples[i,:])
        allMark[i] = testFunc.isOK(allSamples[i,:])

    #空间缩减，保留可行域外围1.1倍区域的超立方空间，避免因点数过多引起的计算冗余
    space_min,space_max = FeasibleSpace(svm,0.1,[100,100])
    x, y = np.mgrid[space_min[0]:space_max[0]:100j, space_min[1]:space_max[1]:100j]

    #剔除搜索区域之外的样本点

    l = []
    for i in range(allSamples.shape[0]):
        if allSamples[i,0]<space_min[0] or allSamples[i,1]<space_min[1] \
        or allSamples[i,0]>space_max[0] or allSamples[i,1]>space_max[1]:
            l.append(i)


    samples = np.delete(allSamples,l,axis=0)
    value = np.delete(allValue,l)
    mark = np.delete(allMark,l)

    #建立响应面
    kriging = Kriging()
    # theta = [32.22662213, 18.59361027]
    # kriging.fit(samples, value, space_min, space_max, theta)
    
    print('正在优化theta参数....')    
    kriging.fit(samples, value, space_min, space_max)
    theta = kriging.optimize(10000,root_path+'/ADE_theta.txt')



    # 搜索kriging模型在可行域中的最优值
    def kriging_optimum(x):
        y = kriging.get_Y(x)
        penaltyItem = penalty*min(0,svm.transform(x))
        return y-penaltyItem

    print('搜索kriging模型在约束区间的最优值.....')
    ade = ADE(space_min, space_max, 100, 0.5, kriging_optimum,True)
    opt_ind = ade.evolution(maxGen=100000)
    kriging.optimumLocation = opt_ind.x
    kriging.optimum = kriging.get_Y(opt_ind.x)

    #目标函数是EI函数和约束罚函数的组合函数
    def EI_optimum(x):
        ei = kriging.EI(x)
        penaltyItem = penalty*min(0,svm.transform(x))
        return ei + penaltyItem

    def Varience_optimum(x):
        s = kriging.get_S(x)
        penaltyItem = penalty*min(0,svm.transform(x))
        return s + penaltyItem

    iterNum = 100    #加点数目
    preValue = np.zeros_like(x)
    EI_Value = np.zeros_like(x)
    varience = np.zeros_like(x)

    maxEI_threshold = 0.0001
    optimum_threshold = 0.0001
    smallestDistance = 0.01
    lastOptimum = None

    for k in range(iterNum):
        #遍历响应面
        print('正在遍历响应面...')
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                a=[x[i, j], y[i, j]]
                if (svm.transform(a)<0):
                    preValue[i,j] = 0
                    varience[i,j] = 0
                    EI_Value[i,j] = 0      
                else:         
                    preValue[i,j] = kriging.get_Y(a)
                    varience[i,j] = kriging.get_S(a)
                    EI_Value[i,j] = kriging.EI(a)

        path1 = root_path+'/Kriging_Predicte_Model_%d.txt'%k
        writeFile([x,y,preValue],[samples,value],path1)
        path2 = root_path+'/Kriging_Varience_Model_%d.txt'%k
        writeFile([x,y,varience],[samples,value],path2)
        path3 = root_path+'/Kriging_EI_Model_%d.txt'%k
        writeFile([x,y,EI_Value],[samples,value],path3)

        print('\n第%d轮加点.........'%k)
        #每轮加点为方差最大值，EI函数最大值

        # 不建议加入最优值，因为最优值与原有采样点的距离过于接近了。
        # nextSample = kriging.optimumLocation

        print('搜索EI函数在约束区间的最优值.....')
        ade = ADE(space_min, space_max, 100, 0.5, EI_optimum,False)
        opt_ind = ade.evolution(maxGen=100000)
        nextSample = opt_ind.x
        # nextSample = np.vstack((nextSample,opt_ind.x))
        maxEI = EI_optimum(opt_ind.x)

        print('搜索方差在约束区间的最优值.....')
        ade = ADE(space_min,space_max,100,0.5,Varience_optimum,False)
        opt_ind = ade.evolution(10000,0.8)
        nextSample = np.vstack((nextSample,opt_ind.x))

        #如果加点过于逼近，只选择一个点
        nextSample = filterSamples(nextSample,samples,smallestDistance)

        #判定终止条件
        if k == 0:
            lastOptimum = kriging.optimum
        else:
            # 当MaxEI小于EI门限值说明全局已经没有提升可能性
            if maxEI < maxEI_threshold:
                print('EI全局最优值小于%.5f,计算终止'%maxEI_threshold)
                break
            else:
                print('EI全局最优值%.5f'%maxEI)

            #以最优值提升为终止条件极容易导致算法过早停止
            # # 当全局最优值两轮变化小于最优值门限
            # if abs(lastOptimum-kriging.optimum) < optimum_threshold:
            #     print('kriging模型全局最优值的提升小于%.5f，计算终止'%optimum_threshold)
            #     break
            # else:
            #     print('kriging模型全局最优值的提升%.5f'%(abs(lastOptimum-kriging.optimum)))
            #     lastOptimum = kriging.optimum
            
            # 当加点数目为0，说明新加点与原有点的距离过近
            if nextSample.shape[0] == 0:
                print('新加点的数目为0 ，计算终止')
                break
            else:
                print('本轮加点数目%d'%nextSample.shape[0])



        # 检查新样本点是否满足约束，并检查SVM判定结果。
        # 如果SVM判定失误，重新训练SVM模型
        # 如果SVM判定正确，但是采样点不满足约束，惩罚系数×2。
        nextSampleNum = nextSample.shape[0]
        nextValue = np.zeros(nextSampleNum)
        nextFuncMark = np.zeros(nextSampleNum)
        nextSVMMark = np.zeros(nextSampleNum)
        for i in range(nextSampleNum):
            nextValue[i] = testFunc.aim(nextSample[i,:])
            nextFuncMark[i] = testFunc.isOK(nextSample[i,:])
            nextSVMMark[i] = svm.transform(nextSample[i,:])



        samples = np.vstack((samples,nextSample))
        value = np.append(value,nextValue)
        mark = np.append(mark,nextFuncMark)

        allSamples = np.vstack((allSamples,nextSample))
        allValue = np.append(allValue,nextValue)
        allMark = np.append(allMark, nextFuncMark)

        for i in range(nextSampleNum):
            if (nextFuncMark[i] == -1 and nextSVMMark[i] > 0) or (nextFuncMark[i] == 1 and nextSVMMark[i] < 0):
                print('新采样点的计算结果与SVM判定不符，重新训练SVM模型.......')
                svm.fit(samples,mark,500000)
                svm.show()

                #不建议多次设定搜索区域，首先SVM的外围需要一定的负样本来限定超平面，同时kriging模型
                #也需要负样本来评判目标函数在边界处的取值。反复设定搜索区域的确会减少点的数目，但约束外侧
                #过少的采样点会使后续的预测恶化。

                # #设定搜索区域
                # space_min,space_max = FeasibleSpace(svm,0.1)
                # x, y = np.mgrid[space_min[0]:space_max[0]:100j, space_min[1]:space_max[1]:100j]

                # #剔除搜索区域之外的样本点

                # l = []
                # for i in range(allSamples.shape[0]):
                #     if allSamples[i,0]<space_min[0] or allSamples[i,1]<space_min[1] \
                #     or allSamples[i,0]>space_max[0] or allSamples[i,1]>space_max[1]:
                #         l.append(i)


                # samples = np.delete(allSamples,l,axis=0)
                # value = np.delete(allValue,l)
                # mark = np.delete(allMark,l)
                break

            if nextFuncMark[i] == -1 and nextSVMMark[i] < 0:
                print('新采样点位于违反约束区域，惩罚系数乘2')
                penalty *= 1.1
        
        print('正在优化theta参数....')    
        kriging.fit(samples, value, space_min, space_max)
        theta = kriging.optimize(10000,root_path+'/ADE_theta.txt')

        print('搜索kriging模型在约束区间的最优值.....')       
        ade = ADE(space_min, space_max, 100, 0.5, kriging_optimum ,True)
        opt_ind = ade.evolution(maxGen=100000)
        kriging.optimumLocation = opt_ind.x
        kriging.optimum = kriging.get_Y(opt_ind.x)

        Data = np.hstack((samples,value.reshape((-1,1)),mark.reshape((-1,1))))
        np.savetxt(root_path+'/优化结果.txt',Data,delimiter='\t')



    print('全局最优值:',kriging.optimum)
    print('全局最优值坐标:',kriging.optimumLocation)

class SKCO(object):
    '''基于SVM和kriging的含约束优化方法\n
    input :\n
    func : 求解问题实例。结构可参考本文件中的TestFunction_G4，必须包含目标函数，约束，自变量区间等数据\n
    logPath : 日志文件存储位置，用于存储计算中产生的数据，日志'''
    def __init__(self,func,logPath):
        '''初始化函数'''
        self.f = func
        self.logPath = logPath
        import os 
        if not os.path.exists(logPath):
            os.makedirs(logPath) 
        #采样点
        self.samples = None
        self.value = None
        self.mark = None

    def Step_A(self,initSampleNum = 100,auxiliarySampleNum = 10):
        '''初步搜索设计空间\n
        input : \n
        initSampleNum : 整型，初始采样点数目\n
        auxiliarySampleNum : 整型，附加采样点数目\n'''

        #生成采样点
        lh=LatinHypercube(self.f.dim,initSampleNum,self.f.min,self.f.max)
        samples=lh.realSamples
        np.savetxt(self.logPath+'/InitSamples.txt',samples,delimiter=',')

        # samples = np.loadtxt(self.logPath+'/InitSamples.txt',delimiter=',')

        value=np.zeros(initSampleNum)
        for i in range(initSampleNum):
            value[i]=self.f.aim(samples[i,:])
        
        #建立响应面
        kriging = Kriging()
        kriging.fit(samples, value, self.f.min, self.f.max)
        
        print('正在优化theta参数...')
        theta = kriging.optimize(10000,self.logPath+'/theta优化种群数据.txt')
        # theta = [0.04594392,0.001,0.48417354,0.001,0.02740766]

        for k in range(auxiliarySampleNum):
            print('第%d次加点...'%(k+1))
            nextSample = kriging.nextPoint_Varience()
            samples = np.vstack([samples,nextSample])
            value = np.append(value,self.f.aim(nextSample))
            kriging.fit(samples, value, self.f.min, self.f.max, theta)
            # kriging.optimize(100)

        #检测样本点中是否有可行解，如果没有继续加点
        mark = np.zeros(samples.shape[0])
        for i in range(samples.shape[0]):
            mark[i] = self.f.isOK(samples[i,:])
        
        if np.sum(mark==1)>0:
            value = value.reshape((-1,1))
            mark = mark.reshape((-1,1))
            storeData = np.hstack((samples,value,mark))
            np.savetxt(self.logPath+'/A_Samples.txt',storeData)
            return
        else:
            print('在所有样本中未能发现可行域，继续加点...')

        
        i = 0
        while mark[-1]==-1:
            i += 1
            print('第%d次加点...'%(auxiliarySampleNum+i))
            nextSample = kriging.nextPoint_Varience()
            samples = np.vstack([samples,nextSample])
            value = np.append(value,self.f.aim(nextSample))
            mark = np.append(mark,self.f.isOK(nextSample))
            kriging.fit(samples, value, self.f.min, self.f.max, theta)
            # kriging.optimize(100)

        value = value.reshape((-1,1))
        mark = mark.reshape((-1,1))
        storeData = np.hstack((samples,value,mark))
        np.savetxt(self.logPath+'/A_Samples.txt',storeData)

    def Step_B(self,T0_list,T1_list):
        '''
        应用SVM分割设计空间，并按照T1_list中的参数设置优化超平面
        '''
        if len(T0_list) != len(T1_list):
            raise ValueError('T0列表与T1列表数目不相符')

        #理论分割函数
        f = self.f
        data = np.loadtxt(self.logPath+'/A_Samples.txt')
        samples = data[:,0:f.dim]
        mark = data[:,f.dim+1]
        Kernal_Gau = lambda x,y:np.exp((-np.linalg.norm(x-y)**2)/80)
        Kernal_Poly = lambda x,y:(np.dot(x,y)+1)**7
        svm=SVM(5,kernal=Kernal_Gau,path = self.logPath,fileName='SVM_Step_B.txt')
        print('训练初始支持向量机...')
        svm.fit(samples,mark,maxIter=20000,maxAcc=1.1)
        test28(svm=svm)

        #记录每轮加点的数目
        pointNum = np.zeros(len(T1_list)+1)
        pointNum[0] = samples.shape[0]

        for k in range(len(T1_list)):
            print('\n第%d轮加点...'%(k+1))
            new_x = svm.infillSample4(T0_list[k],T1_list[k],f.min,f.max,[12,6,9,9,9])
            if new_x is None:
                print('当T1设置为%.2f时，加点数目为0'%T1_list[k])
                pointNum[k+1] = samples.shape[0]
                continue
            else:
                num = new_x.shape[0]

            new_mark = np.zeros(num)
            for i in range(num):
                new_mark[i] = f.isOK(new_x[i,:])
            samples = np.vstack((samples,new_x))
            mark = np.append(mark,new_mark)
            print('训练支持向量机...')
            svm.fit(samples,mark,20000,maxAcc=1.1)
            
            test28(svm=svm)
            pointNum[k+1] = samples.shape[0]
            print('本轮加点数目：%d'%pointNum[k+1])
              

        value = np.zeros(samples.shape[0])
        for i in range(samples.shape[0]):
            value[i] = f.aim(samples[i,:])
        value = value.reshape((-1,1))
        mark = mark.reshape((-1,1))
        storeData = np.hstack((samples,value,mark))
        np.savetxt(self.logPath+'/B_Samples.txt',storeData)        

        print('样本点数目：')
        print(pointNum)
        print('加点结束')    

    def Step_C(self):
        #违反约束的惩罚系数
        #惩罚系数必须足够大，足以弥补EI函数与y之间的数量差距
        penalty = 10000000000000

        # 加载支持向量机
        Kernal_Gau = lambda x,y:np.exp((-np.linalg.norm(x-y)**2)/90)
        Kernal_Poly = lambda x,y:(np.dot(x,y)+1)**9
        svm=SVM(5,kernal=Kernal_Gau,path =self.logPath,fileName='SVM_Step_C.txt' )
        
        # 提取已采样样本的坐标，值，是否违反约束的标志
        testFunc = TestFunction_G4()

        data = np.loadtxt(self.logPath+'/B_Samples.txt')
        allSamples = data[:,0:testFunc.dim]
        allValue = data[:,testFunc.dim]
        allMark = data[:,testFunc.dim+1]     

        print('训练初始支持向量机...')
        svm.fit(allSamples,allMark,30000,maxAcc=1.1)
        test28(svm) 
        # svm.retrain(self.logPath+'/SVM_Step_B.txt',maxIter=10 ,maxAcc=1.1)

        # #空间缩减，保留可行域外围1.1倍区域的超立方空间，避免因点数过多引起的计算冗余
        # space_min,space_max = FeasibleSpace(svm,[12,6,9,9,9],0.1)

        # #为了防止过拟合，支持向量机的训练并不彻底。所以也存在正例被判定为反例的情况。
        # # 导致SVM的正例区域不一定完整包含可行域，需要提取当前采样点的正例的空间，两个空间求并

        # allSamples_pos = allSamples[allMark==1]
        # add_space_min = np.min(allSamples_pos,0)
        # add_space_max = np.max(allSamples_pos,0)        
        # add_space_max = (add_space_max-add_space_min)*0.1+add_space_max
        # add_space_min = add_space_min-(add_space_max-add_space_min)*0.1
        # for i in range(testFunc.dim):
        #     if space_min[i]>add_space_min[i]:
        #         space_min[i] = add_space_min[i]
        #     if space_max[i]<add_space_max[i]:
        #         space_max[i] = add_space_max[i]

        # #与函数取值空间比对，求交集
        # for i in range(testFunc.dim):
        #     if space_min[i]<testFunc.min[i]:
        #         space_min[i] = testFunc.min[i]
        #     if space_max[i]>testFunc.max[i]:
        #         space_max[i] = testFunc.max[i]

        space_min = testFunc.min
        space_max = testFunc.max
        #剔除搜索区域之外的样本点

        l = []
        for i in range(allSamples.shape[0]):
            for j in range(testFunc.dim):
                if allSamples[i,j]<space_min[j] or allSamples[i,j]>space_max[j]:
                    l.append(i)
                    break

        samples = np.delete(allSamples,l,axis=0)
        value = np.delete(allValue,l)
        mark = np.delete(allMark,l)

        #建立响应面
        kriging = Kriging()
        # theta = [0.234505671,0.001,1.84605460,0.001,0.145689767]
        # kriging.fit(samples, value, space_min, space_max, theta)
        
        print('正在优化theta参数....')    
        kriging.fit(samples, value, space_min, space_max)
        theta = kriging.optimize(10000,self.logPath+'/ADE_theta.txt')

        # 搜索kriging模型在可行域中的最优值
        def kriging_optimum(x):
            y = kriging.get_Y(x)
            penaltyItem = penalty*min(0,svm.transform(x))
            return y-penaltyItem


        #kriging的global_optimum函数只能找到全局最优，而不是可行域最优
        print('搜索kriging模型在约束区间的最优值.....')
        ade = ADE(space_min, space_max, 200, 0.5, kriging_optimum,True)
        opt_ind = ade.evolution(maxGen=5000)
        kriging.optimumLocation = opt_ind.x
        kriging.optimum = kriging.get_Y(opt_ind.x)
        print('SVM对最优值的判定结果%.4f'%svm.transform(kriging.optimumLocation))         

        # testX = [78,34.6102252,30.98470067,29.68978243,28.85514208]
        # print('SVM对最优值的判定结果%.4f'%svm.transform(testX))        
        # kriging.optimumLocation = testX
        # kriging.optimum = kriging.get_Y(testX)

        #目标函数是EI函数和约束罚函数的组合函数
        def EI_optimum(x):
            ei = kriging.EI(x)
            penaltyItem = penalty*min(0,svm.transform(x))
            return ei + penaltyItem

        def Varience_optimum(x):
            s = kriging.get_S(x)
            penaltyItem = penalty*min(0,svm.transform(x))
            return s + penaltyItem

        iterNum = 100    #迭代次数
        maxEI_threshold = 0.0001
        smallestDistance = 0.01


        for k in range(iterNum):
            print('\n第%d轮加点.........'%k)
            #每轮加点为方差最大值，EI函数最大值

            print('搜索EI函数在约束区间的最优值.....')
            ade = ADE(space_min, space_max, 200, 0.5, EI_optimum,False)
            opt_ind = ade.evolution(maxGen=5000)
            nextSample = opt_ind.x
            maxEI = EI_optimum(opt_ind.x)

            print('搜索方差在约束区间的最优值.....')
            ade = ADE(space_min,space_max,200,0.5,Varience_optimum,False)
            opt_ind = ade.evolution(5000,0.8)
            nextSample = np.vstack((nextSample,opt_ind.x))

            #如果加点过于逼近，只选择一个点
            nextSample = filterSamples(nextSample,samples,smallestDistance)

            #判定终止条件

            # 当MaxEI小于EI门限值说明全局已经没有提升可能性
            if maxEI < maxEI_threshold:
                print('EI全局最优值小于%.5f,计算终止'%maxEI_threshold)
                break
            else:
                print('EI全局最优值%.5f'%maxEI)
            
            # 当加点数目为0，说明新加点与原有点的距离过近
            if nextSample.shape[0] == 0:
                print('新加点的数目为0 ，计算终止')
                break
            else:
                print('本轮加点数目%d'%nextSample.shape[0])

            # 检查新样本点是否满足约束，并检查SVM判定结果。
            # 如果SVM判定失误，重新训练SVM模型
            # 如果SVM判定正确，但是采样点不满足约束，惩罚系数×2。
            nextSampleNum = nextSample.shape[0]
            nextValue = np.zeros(nextSampleNum)
            nextFuncMark = np.zeros(nextSampleNum)
            nextSVMMark = np.zeros(nextSampleNum)
            for i in range(nextSampleNum):
                nextValue[i] = testFunc.aim(nextSample[i,:])
                nextFuncMark[i] = testFunc.isOK(nextSample[i,:])
                nextSVMMark[i] = svm.transform(nextSample[i,:])

            samples = np.vstack((samples,nextSample))
            value = np.append(value,nextValue)
            mark = np.append(mark,nextFuncMark)

            allSamples = np.vstack((allSamples,nextSample))
            allValue = np.append(allValue,nextValue)
            allMark = np.append(allMark, nextFuncMark)

            for i in range(nextSampleNum):
                if (nextFuncMark[i] == -1 and nextSVMMark[i] > 0) or (nextFuncMark[i] == 1 and nextSVMMark[i] < 0):
                    print('新采样点的计算结果与SVM判定不符，重新训练SVM模型.......')
                    svm.fit(samples,mark,30000,maxAcc=1.1)
                    test28(svm)
                    break

                if nextFuncMark[i] == -1 and nextSVMMark[i] < 0:
                    print('新采样点位于违反约束区域，惩罚系数乘2')
                    penalty *= 1.1
            
            print('正在优化theta参数....')    
            kriging.fit(samples, value, space_min, space_max,theta)

            print('搜索kriging模型在约束区间的最优值.....')       
            ade = ADE(space_min, space_max, 200, 0.5, kriging_optimum ,True)
            opt_ind = ade.evolution(maxGen=5000)
            kriging.optimumLocation = opt_ind.x
            kriging.optimum = kriging.get_Y(opt_ind.x)

            Data = np.hstack((allSamples,allValue.reshape((-1,1)),allMark.reshape((-1,1))))
            np.savetxt(self.logPath+'/全部样本点.txt',Data,delimiter='\t')

        print('全局最优值:',kriging.optimum)
        print('全局最优值坐标:',kriging.optimumLocation)        

    def Test_SVM_Kernal(self):
        #理论分割函数
        f = self.f   

        pointNum = 1
        dimNum = [12,6,9,9,9]
        weight = np.zeros(f.dim)
        for i in range(f.dim):
            pointNum *= dimNum[i]
            weight[i] = (f.max[i]-f.min[i])/(dimNum[i]-1)   

        points = np.zeros((pointNum,f.dim))    
        points_mark = np.zeros(pointNum) 
        for i in range(f.dim):
            points[:,i] = f.min[i]

        for i in range(pointNum):
            index = i
            for j in range(f.dim):
                points[i,j] += index%dimNum[j]*weight[j]
                index = index // dimNum[j]
                if index == 0:
                    break       
            points_mark[i] = f.isOK(points[i,:])

        data = np.loadtxt(self.logPath+'/B_Samples.txt')
        samples = data[:,0:f.dim]
        mark = data[:,f.dim+1]

        # for exponent in range(5,11):
        #     Kernal_Poly= lambda x,y:(np.dot(x,y)+1)**exponent
        #     svm=SVM(5,kernal=Kernal_Poly,path = self.logPath,fileName='SVM_Argument_exponent_%d.txt'%exponent)
        #     print('多项式核函数指数:%d,开始训练支持向量机...'%exponent)
        #     svm.fit(samples,mark,maxIter=10000,maxAcc=0.9) 

        #     TP = 0
        #     FN = 0
        #     TN = 0
        #     FP = 0    

        #     for i in range(pointNum):         
        #         y_ture = points_mark[i]
        #         y_pred = svm.transform(points[i,:])
        #         if y_ture>0 and y_pred>0:
        #             TP += 1
        #         elif y_ture>0 and y_pred<0:
        #             FN += 1
        #         elif y_ture<0 and y_pred>0:
        #             FP += 1
        #         elif y_ture<0 and y_pred<0: 
        #             TN += 1           

        #     E = (FP + FN)/(pointNum)
        #     acc = 1-E
        #     if TP == 0:
        #         P = 0
        #         R = 0
        #         F1 = 0
        #     else:
        #         P = TP/(TP+FP) 
        #         R = TP/(TP+FN)
        #         F1 = 2*P*R/(P+R)
        #     print('........................')
        #     print('核函数为多项式核函数，指数:%d'%exponent)
        #     print('样本点总数目:%d'%pointNum)
        #     print('正例数目:%d'%int(TP+FN))
        #     print('反例数目:%d'%int(TN+FP))
        #     print('真正例（将正例判定为正例）:%d'%TP)
        #     print('假正例（将反例判定为正例）:%d'%FP)
        #     print('真反例（将反例判定为反例）:%d'%TN)
        #     print('假反例（将正例判定为反例）:%d'%FN)
        #     print('错误率:%.4f'%E)
        #     print('精度:%.4f'%acc)
        #     print('查准率:%.4f'%P)
        #     print('查全率:%.4f'%R)
        #     print('F1:%.4f'%F1)

        #用同样的方法检测高斯核函数
        for exponent in range(4,11):
            Kernal_Gau = lambda x,y:np.exp((-np.linalg.norm(x-y)**2)/(10*exponent))
            svm=SVM(5,kernal=Kernal_Gau,path = self.logPath,fileName='SVM_Argument_Gaussian_%d.txt'%exponent)
            print('核函数为高斯核函数,开始训练支持向量机...')
            svm.fit(samples,mark,maxIter=20000,maxAcc=1.1) 

            TP = 0
            FN = 0
            TN = 0
            FP = 0    

            for i in range(pointNum):
                y_ture = points_mark[i]
                y_pred = svm.transform(points[i,:])
                if y_ture>0 and y_pred>0:
                    TP += 1
                elif y_ture>0 and y_pred<0:
                    FN += 1
                elif y_ture<0 and y_pred>0:
                    FP += 1
                elif y_ture<0 and y_pred<0: 
                    TN += 1           

            E = (FP + FN)/(pointNum)
            acc = 1-E
            if TP == 0:
                P = 0
                R = 0
                F1 = 0
            else:
                P = TP/(TP+FP) 
                R = TP/(TP+FN)
                F1 = 2*P*R/(P+R)
            print('........................')
            print('核函数为高斯核函数,2*sigma^2=%d'%(exponent*10))
            print('样本点总数目:%d'%pointNum)
            print('正例数目:%d'%int(TP+FN))
            print('反例数目:%d'%int(TN+FP))
            print('真正例（将正例判定为正例）:%d'%TP)
            print('假正例（将反例判定为正例）:%d'%FP)
            print('真反例（将反例判定为反例）:%d'%TN)
            print('假反例（将正例判定为反例）:%d'%FN)
            print('错误率:%.4f'%E)
            print('精度:%.4f'%acc)
            print('查准率:%.4f'%P)
            print('查全率:%.4f'%R)
            print('F1:%.4f'%F1)

    def Test_SVM_IterNum(self):
        '''测试迭代次数对支持向量机的影响'''
        f = self.f   

        pointNum = 1
        dimNum = [12,6,9,9,9]
        weight = np.zeros(f.dim)
        for i in range(f.dim):
            pointNum *= dimNum[i]
            weight[i] = (f.max[i]-f.min[i])/(dimNum[i]-1)   

        points = np.zeros((pointNum,f.dim))    
        points_mark = np.zeros(pointNum) 
        for i in range(f.dim):
            points[:,i] = f.min[i]

        for i in range(pointNum):
            index = i
            for j in range(f.dim):
                points[i,j] += index%dimNum[j]*weight[j]
                index = index // dimNum[j]
                if index == 0:
                    break       
            points_mark[i] = f.isOK(points[i,:])

        data = np.loadtxt(self.logPath+'/A_Samples.txt')
        samples = data[:,0:f.dim]
        mark = data[:,f.dim+1]

        Kernal_Poly = lambda x,y:(np.dot(x,y)+1)**7
        svm=SVM(1,kernal=Kernal_Poly,path = self.logPath,fileName='SVM_Argument.txt')
        print('训练初始支持向量机...')
        svm.fit(samples,mark,maxIter=0)

        step_list = [100,100,100,100,100,100,100,100,100,100,\
            1000,1000,1000,1000,1000,1000,1000,1000,1000,\
            10000,10000,10000,10000,10000,10000,10000,10000,10000]
        iterNum = 0
        for step in step_list:
            print('训练支持向量机...')
            svm.retrain(self.logPath+'/SVM_Argument.txt',maxIter=step)
            iterNum += step 

            TP = 0
            FN = 0
            TN = 0
            FP = 0    

            for i in range(pointNum):
                y_ture = points_mark[i]
                y_pred = svm.transform(points[i,:])
                if y_ture>0 and y_pred>0:
                    TP += 1
                elif y_ture>0 and y_pred<0:
                    FN += 1
                elif y_ture<0 and y_pred>0:
                    FP += 1
                elif y_ture<0 and y_pred<0: 
                    TN += 1           

            E = (FP + FN)/(pointNum)
            acc = 1-E
            if TP == 0:
                P = 0
                R = 0
            else:
                P = TP/(TP+FP) 
                R = TP/(TP+FN)
            F1 = 2*P*R/(P+R)
            print('........................')
            print('迭代次数：%d'%iterNum)
            print('样本点总数目:%d'%pointNum)
            print('正例数目:%d'%int(TP+FN))
            print('反例数目:%d'%int(TN+FP))
            print('真正例（将正例判定为正例）:%d'%TP)
            print('假正例（将反例判定为正例）:%d'%FP)
            print('真反例（将反例判定为反例）:%d'%TN)
            print('假反例（将正例判定为反例）:%d'%FN)
            print('错误率:%.4f'%E)
            print('精度:%.4f'%acc)
            print('查准率:%.4f'%P)
            print('查全率:%.4f'%R)
            print('F1:%.4f'%F1)      


def SKCO_test():
    f = TestFunction_G4()
    skco = SKCO(f,'./Data/G4函数测试2')
    # skco.Step_A(51,20)
    # skco.Test_SVM_Kernal()    
    # skco.Test_SVM_IterNum()

    # skco.Step_B([10000,10000],[10,10])

    skco.Step_C()

if __name__=='__main__':
    SKCO_test()

