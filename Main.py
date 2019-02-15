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

def FeasibleSpace(svm, edgeRatio):
    '''
    提取包含支持向量机的可行域（正例）的超立方空间。\n
    input : \n
    svm : SVM类实例\n
    edgeRatio : 取值0~1的浮点数，表示一个维度上，可行域向外围延展的区域与可行域在该维度投影区域的比例\n
    output : \n
    space_min,space_max,第一项是超立方空间的下限，第二项是超立方空间的上限\n'''
    dim = svm.x.shape[1]

    space_min = np.zeros(dim)
    space_max = np.zeros(dim)
    for i in range(dim):
        space_min[i] = min(svm.x[:,i])
        space_max[i] = max(svm.x[:,i])
    pointNum = 100

    coordinate = np.zeros((pointNum**dim,dim))
    mark = np.zeros(pointNum**dim)    
    for i in range(dim):
        coordinate[:,i]+=space_min[i]

    for n in range(pointNum**dim):
        index = n
        for m in range(dim):
           i = index%pointNum
           coordinate[n,m]+=(space_max[m]-space_min[m])/(pointNum-1)*i
           index =  index//pointNum
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
    space_min,space_max = FeasibleSpace(svm,0.1)
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



if __name__=='__main__':
    stepC_1()
