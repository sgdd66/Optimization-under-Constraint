#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***************************************************************************
# Copyright (c) 2019 西安交通大学
# All rights reserved
# 
# 文件名称：Main.py
# 
# 摘    要：针对G4问题的skco方法
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
from DOE import LatinHypercube
import numpy as np
from ADE import ADE
from sklearn import svm as SVM_SKLearn
import matplotlib.pyplot as plt

class TestFunction_G4_Simple(object):
    '''
    测试函数G4 \n
    变量维度 : 5\n
    搜索空间 : l=(27,27,29),u=(45,45,40),li<xi<ui,i=1...5\n
    全局最优值 : x* =  (29.996,45,36.7758),f(x*) = -30665.539
    '''
    def __init__(self):
        '''建立目标和约束函数'''

        self.dim = 3
        self.l = [27,27,29]
        self.u = [45,45,40]

        self.optimum = [29.9952560256815985,45,36.7758129057882073]

        self.data = np.loadtxt('./Data/G4简化函数测试1/G4简化函数空间0.txt')

    def aim_Matrix(self,M):
        A = 5.3578547*M[:,0]**2+0.8356891*78*M[:,2]+37.293239*78-40792.141
        return A

    def aim(self,x):
        A = 5.3578547*x[0]**2+0.8356891*78*x[2]+37.293239*78-40792.141
        return A

    def isOK_Matrix(self,M):
        '''检查样本点x是否违背约束，是返回-1，否返回1\n
        input : \n
        x : 样本点，一维向量\n
        output : \n
        mark : int，-1表示违反约束，1表示不违反约束\n'''
        if M.shape[1] != self.dim:
            raise ValueError('isOK：参数维度与测试函数维度不匹配')
        
        if np.sum(M<self.l)>0 or np.sum(M>self.u)>0:
            raise ValueError('TestFunction_G16: 参数已超出搜索空间')     

        u = 85.334407+0.0056858*33*M[:,2]+0.0006262*78*M[:,1]-0.0022053*M[:,0]*M[:,2]
        v = 80.51249+0.0071317*33*M[:,2]+0.0029955*78*33+0.0021813*M[:,0]**2
        w = 9.300961+0.0047026*M[:,0]*M[:,2]+0.0012547*78*M[:,0]+0.0019085*M[:,0]*M[:,1]

        #约束函数，小于等于0为满足约束
        g = np.zeros((M.shape[0],6))
        g[:,0] = u-92
        g[:,1] = -u
        g[:,2] = v-110
        g[:,3] = -v+90
        g[:,4] = w-25
        g[:,5] = -w+20
        mark = np.sum(g>0,axis=1)
        mark[np.where(mark>0)] = -1
        mark[np.where(mark==0)] = 1

        return mark

    def isOK(self,x):
        if len(x) != self.dim:
            raise ValueError('isOK：参数维度与测试函数维度不匹配')
          

        u = 85.334407+0.0056858*33*x[2]+0.0006262*78*x[1]-0.0022053*x[0]*x[2]
        v = 80.51249+0.0071317*33*x[2]+0.0029955*78*33+0.0021813*x[0]**2
        w = 9.300961+0.0047026*x[0]*x[2]+0.0012547*78*x[0]+0.0019085*x[0]*x[1]

        #约束函数，小于等于0为满足约束
        g = np.zeros(6)
        g[0] = u-92
        g[1] = -u
        g[2] = v-110
        g[3] = -v+90
        g[4] = w-25
        g[5] = -w+20
        mark = np.sum(g>0)

        if mark>0:
            return -1
        else:
            return 1


    def report(self,svm):
        '''比较SVM与实际分类差异'''
        pointNum = self.data.shape[0]
        points_mark = self.data[:,self.dim]
        points = self.data[:,0:self.dim]
        svm_mark = svm.predict(points)        

        TP = 0
        FN = 0
        TN = 0
        FP = 0    

        points_pos = points_mark==1
        points_neg = ~points_pos

        svm_pos = svm_mark==1
        svm_neg = ~svm_pos


        TP = np.sum(points_pos & svm_pos)
        FP = np.sum(svm_pos & points_neg)
        TN = np.sum(points_neg & svm_neg)
        FN = np.sum(svm_neg & points_pos)    

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
        print('样本点总数目:%d'%pointNum)
        print('正例数目:%d'%int(TP+FN))
        print('反例数目:%d'%int(TN+FP))
        print('真正例（将正例判定为正例）:%d'%TP)
        print('假正例（将反例判定为正例）:%d'%FP)
        print('真反例（将反例判定为反例）:%d'%TN)
        print('假反例（将正例判定为反例）:%d'%FN)
        # print('错误率:%.4f'%E)
        print('精度:%.8f'%acc)
        print('查准率:%.8f'%P)
        print('查全率:%.8f'%R)
        print('F1:%.8f'%F1)

        x = self.optimum
        # print('实际最优值坐标：',x)
        # print('实际最优值:%.6f'%self.aim(x))
        print('SVM对实际最优值判定:%.8f'%svm.decision_function([x]))

    def TestData(self):   

        pointNum = 1
        dimNum = [50,50,50]
        weight = np.zeros(self.dim)
        for i in range(self.dim):
            pointNum *= dimNum[i]
            weight[i] = (self.u[i]-self.l[i])/(dimNum[i]-1)   
        maxPointNum = 10000000.0
        iterNum = int(np.ceil(pointNum/maxPointNum))
        for fileIndex in range(iterNum):
            if fileIndex == iterNum-1:
                iterPointNum = int(pointNum%maxPointNum)
            else:
                iterPointNum = maxPointNum

            points = np.zeros((iterPointNum,self.dim))    
            points_mark = np.zeros(iterPointNum) 

            for i in range(self.dim):
                points[:,i] = self.l[i]

            for i in range(iterPointNum):
                index = i+fileIndex*maxPointNum
                for j in range(self.dim):
                    points[i,j] += index%dimNum[j]*weight[j]
                    index = index // dimNum[j]
                    if index == 0:
                        break       
                
            points_mark = self.isOK_Matrix(points)
            points_mark = points_mark.reshape((-1,1))
            data = np.hstack((points,points_mark))
            
            np.savetxt('./Data/G4简化函数测试1/G4简化函数空间%d.txt'%fileIndex,data)

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
        mark = np.zeros(initSampleNum)-1
        while np.sum(mark == 1)==0:
            lh=LatinHypercube(self.f.dim,initSampleNum,self.f.l,self.f.u)
            samples=lh.realSamples
            mark = self.f.isOK_Matrix(samples)


        np.savetxt(self.logPath+'/InitSamples.txt',samples,delimiter=',')

        # samples = np.loadtxt(self.logPath+'/InitSamples.txt',delimiter=',')

        value = self.f.aim_Matrix(samples)
        
        #建立响应面
        theta = [6.39935517, 0.663649334, 14.2249506, 6.65649918, 0.001]
        kriging = Kriging()
        kriging.fit(samples, value, self.f.l, self.f.u,theta)
        
        # print('正在优化theta参数...')
        # theta = kriging.optimize(10000,self.logPath+'/theta优化种群数据.txt')

        for k in range(auxiliarySampleNum):
            print('第%d次加点...'%(k+1))
            nextSample = kriging.nextPoint_Varience()
            samples = np.vstack([samples,nextSample])
            value = np.append(value,self.f.aim(nextSample))
            kriging.fit(samples, value, self.f.l, self.f.u, theta)
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
            kriging.fit(samples, value, self.f.l, self.f.u, theta)
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

        # svm=SVM_SKLearn.SVC(C=1000,kernel='linear')
        svm=SVM_SKLearn.SVC(C=1000,kernel='rbf',gamma=0.0005)

        print('训练初始支持向量机...')
        svm.fit(samples,mark)
        f.report(svm)
        
        #记录每轮加点的数目
        pointNum = np.zeros(len(T1_list)+1)
        pointNum[0] = samples.shape[0]

        for k in range(len(T1_list)):
            print('\n第%d轮加点...'%(k+1))
            new_x = self.infillSample2(svm,samples,T0_list[k],T1_list[k])
            if new_x is None:
                print('当T1设置为%.2f时，加点数目为0'%T1_list[k])
                pointNum[k+1] = samples.shape[0]
                continue
            else:
                num = new_x.shape[0]

            new_mark = f.isOK_Matrix(new_x)
            samples = np.vstack((samples,new_x))
            mark = np.append(mark,new_mark)
            print('训练支持向量机...')
            svm.fit(samples,mark)
            f.report(svm)
            
            pointNum[k+1] = samples.shape[0]
            print('本轮样本点总数目：%d'%pointNum[k+1])
              

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
        svm=SVM_SKLearn.SVC(C=1000,kernel='rbf',gamma=0.0005)
        
        # 提取已采样样本的坐标，值，是否违反约束的标志
        testFunc = self.f

        data = np.loadtxt(self.logPath+'/B_Samples.txt')
        samples = data[:,0:testFunc.dim]
        value = data[:,testFunc.dim]
        mark = data[:,testFunc.dim+1]     

        print('训练初始支持向量机...')
        svm.fit(samples,mark)
        self.f.report(svm)
        

        #建立响应面
        kriging = Kriging()

        theta = [28.9228845, 0.001, 0.63095945]
        kriging.fit(samples, value, self.f.l, self.f.u, theta)
        
        # print('正在优化theta参数....')    
        # kriging.fit(samples, value, self.f.l, self.f.u)
        # theta = kriging.optimize(10000,self.logPath+'/ADE_theta.txt')

        # 搜索kriging模型在可行域中的最优值
        def kriging_optimum(x):
            y = kriging.get_Y(x)
            penaltyItem = penalty*min(0,svm.decision_function([x])[0])
            return y-penaltyItem


        #kriging的global_optimum函数只能找到全局最优，而不是可行域最优
        print('搜索kriging模型在约束区间的最优值.....')
        ade = ADE(self.f.l, self.f.u, 200, 0.5, kriging_optimum,True)
        opt_ind = ade.evolution(maxGen=5000)
        kriging.optimumLocation = opt_ind.x
        kriging.optimum = kriging.get_Y(opt_ind.x)
        print('最优值的实际判定结果%.4f'%testFunc.isOK(opt_ind.x))      
        print('最优值的SVM判定结果%.4f'%svm.decision_function([opt_ind.x]))    
        print('最优值实际函数值%.4f'%testFunc.aim(opt_ind.x))            

        #目标函数是EI函数和约束罚函数的组合函数
        def EI_optimum(x):
            ei = kriging.EI(x)
            penaltyItem = penalty*min(0,svm.decision_function([x])[0])
            return ei + penaltyItem

        def Varience_optimum(x):
            s = kriging.get_S(x)
            penaltyItem = penalty*min(0,svm.decision_function([x])[0])
            return s + penaltyItem

        iterNum = 100    #迭代次数
        maxEI_threshold = 0.0001
        smallestDistance = 0.05


        for k in range(iterNum):
            print('\n第%d轮加点.........'%k)
            #每轮加点为方差最大值，EI函数最大值

            print('搜索EI函数在约束区间的最优值.....')
            ade = ADE(self.f.l, self.f.u, 200, 0.5, EI_optimum,False)
            opt_ind = ade.evolution(maxGen=5000)
            nextSample = opt_ind.x
            maxEI = EI_optimum(opt_ind.x)

            while maxEI < 0:
                print('EI函数最优值求解失败,重新求解...')
                ade = ADE(self.f.l, self.f.u, 200, 0.5, EI_optimum,False)
                opt_ind = ade.evolution(maxGen=5000)
                nextSample = opt_ind.x
                maxEI = EI_optimum(opt_ind.x)                
            print('EI函数最优值实际约束判定:%d'%testFunc.isOK(opt_ind.x))

            print('搜索方差在约束区间的最优值.....')
            ade = ADE(self.f.l, self.f.u,200,0.5,Varience_optimum,False)
            opt_ind = ade.evolution(5000,0.8)
            nextSample = np.vstack((nextSample,opt_ind.x))
            print('方差最优值实际约束判定:%d'%testFunc.isOK(opt_ind.x))

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
            if nextSample.shape[0] < 2:
                print('新加点的数目小于2 ，计算终止')
                break
            else:
                print('本轮加点数目%d'%nextSample.shape[0])

            # 检查新样本点是否满足约束，并检查SVM判定结果。
            # 如果SVM判定失误，重新训练SVM模型
            # 如果SVM判定正确，但是采样点不满足约束，惩罚系数×2。
            nextSampleNum = nextSample.shape[0]
            nextValue = np.zeros(nextSampleNum)
            nextFuncMark = np.zeros(nextSampleNum)
            for i in range(nextSampleNum):
                nextValue[i] = testFunc.aim(nextSample[i,:])
                nextFuncMark[i] = testFunc.isOK(nextSample[i,:])

            samples = np.vstack((samples,nextSample))
            value = np.append(value,nextValue)
            mark = np.append(mark,nextFuncMark)

            # 如果只在发现SVM判断错误的前提下重训练，一般只会提高查准率，而不利于查全率的提升。
            # 如果发现最优点满足约束，也应重训练，以增大附近可行区域
            print('训练支持向量机...')
            svm.fit(samples,mark)
            self.f.report(svm)         


            kriging.fit(samples, value,self.f.l, self.f.u, theta)

            print('搜索kriging模型在约束区间的最优值.....')       
            ade = ADE(self.f.l, self.f.u, 200, 0.5, kriging_optimum ,True)
            opt_ind = ade.evolution(maxGen=5000)
            kriging.optimumLocation = opt_ind.x
            kriging.optimum = kriging.get_Y(opt_ind.x)
            print('最优值的实际判定结果%.4f'%testFunc.isOK(kriging.optimumLocation))    
            print('最优值实际函数值%.4f'%testFunc.aim(opt_ind.x))     

            Data = np.hstack((samples,value.reshape((-1,1)),mark.reshape((-1,1))))
            np.savetxt(self.logPath+'/全部样本点.txt',Data,delimiter='\t')


        nextSample = kriging.optimumLocation
        nextValue = testFunc.aim(nextSample)
        nextFuncMark = testFunc.isOK(nextSample)

        samples = np.vstack((samples,nextSample))
        value = np.append(value,nextValue)
        mark = np.append(mark,nextFuncMark)        

        Data = np.hstack((samples,value.reshape((-1,1)),mark.reshape((-1,1))))
        np.savetxt(self.logPath+'/全部样本点.txt',Data,delimiter='\t')   

        while testFunc.isOK(kriging.optimumLocation)==-1:

            print('区间错误,训练支持向量机...')
            svm.fit(samples,mark)
            self.f.report(svm)       

            print('搜索kriging模型在约束区间的最优值.....')  
            kriging.fit(samples, value,self.f.l, self.f.u,theta)     
            ade = ADE(self.f.l, self.f.u, 200, 0.5, kriging_optimum ,True)
            opt_ind = ade.evolution(maxGen=5000)
            kriging.optimumLocation = opt_ind.x
            kriging.optimum = kriging.get_Y(opt_ind.x)   
            print('最优值的实际判定结果%.4f'%testFunc.isOK(kriging.optimumLocation)) 


            nextSample = kriging.optimumLocation
            nextValue = testFunc.aim(nextSample)
            nextFuncMark = testFunc.isOK(nextSample)

            samples = np.vstack((samples,nextSample))
            value = np.append(value,nextValue)
            mark = np.append(mark,nextFuncMark)   

            Data = np.hstack((samples,value.reshape((-1,1)),mark.reshape((-1,1))))
            np.savetxt(self.logPath+'/全部样本点.txt',Data,delimiter='\t')   

        print('全局最优值:',kriging.optimum)
        print('全局最优值坐标:',kriging.optimumLocation)        

    def infillSpace(self,labelNum):
        '''按照指定的参数用采样点密布整个设计空间，返回采样点的坐标\n
        in : \n
        labelNum : 各维度的划分数目\n
        out :\n
        samples : 二维数组，每一行是一个样本点\n
        '''

        #检查各参数维度是否匹配
        dim = self.f.dim
        if dim != len(labelNum):
            raise ValueError('infillSpace:参数维度不匹配')     

        up = self.f.u
        low = self.f.l       

        coordinates = []
        pointNum = 1
        for i in range(dim):
            coordinate = np.linspace(low[i],up[i],labelNum[i])
            coordinates.append(coordinate)
            pointNum*=labelNum[i]

        samples = np.zeros((pointNum,dim))
        for i in range(dim):
            samples[:,i] = low[i]

        for i in range(pointNum):
            ans = i 
            remainder = 0
            for j in range(dim):
                remainder = ans%labelNum[j]
                ans = ans//labelNum[j]
                samples[i,j] = coordinates[j][remainder]
                if ans==0:
                    break
        
        return samples

    def AssemblageDistance(self,A,B):
        '''计算样本集A中每个样本距离样本集B中最近样本的距离\n
        in :\n
        A : 样本集A，二维矩阵，每一行代表一个样本\n
        B : 样本集B，二维矩阵，每一行代表一个样本\n
        out : \n
        distances : 一维向量，数目与样本集A的数目相同，表示样本集A中每个样本距离样本集B中最近样本的距离\n
        '''
        num_A = A.shape[0]
        if A.shape[1]!=B.shape[1]:
            raise ValueError('AssemblageDistance:样本集A与B的维度不匹配')
        
        distances = np.zeros(num_A)
        for i in range(num_A):
            vector = B-A[i,:]
            dis = np.linalg.norm(vector,axis=1)
            distances[i] = np.min(dis)
        
        return distances

    def infillSample4(self,svm,samples,candidateNum,sampleCMaximum,labelNum):
        '''超平面边界加点算法，选取距离超平面最近的数目为candidateNum的样本点，同时用加入样本集C的数目来限制加点密度\n
        in : \n
        svm : 支持向量机实例\n
        samples : 已计算的样本点\n
        candidateNum : Sample_A的初始样本数目\n
        sampleCMaximum : sample_C的最大数目，如果超过此数，加点终止\n
        labelNum : 一维向量，维度与fit函数的x的shape[1]相同，表示产生初始候选集时各维度的划分数目\n
        out : \n
        samples_C : 二维矩阵，每一行是一个样本点。若为None，代表超平面两侧采样点密度满足要求\n
        '''
        #检查参数维度是否匹配
        dim = self.f.dim
        if dim!=len(labelNum):
            raise ValueError('infillSample:参数维度不匹配')


        #生成样本集A，B，C
        samples_A = self.infillSpace(labelNum)
        samples_B = samples
        samples_C = None

        #筛选样本集A，B，保留距离分割超平面距离T0之内的样本点
        num_A = samples_A.shape[0]
        if num_A < candidateNum:
            candidateNum = num_A

        dis_A = svm.decision_function(samples_A)
        
        dis_A_sorted = np.sort(dis_A)
        samples_A = samples_A[np.where(dis_A<=dis_A_sorted[candidateNum-1])] 
        
        #对于样本集B门限约束固定为1.1
        T0_B = 1.1
        dis_B = svm.decision_function(samples_B)
        samples_B = samples_B[np.where(np.abs(dis_B)<T0_B)] 

        if samples_B.shape[0] == 0:
            raise ValueError('infillSample:T0设置过小，区域内没有已采样点')

        #计算样本集A与样本集B的距离
        distances = self.AssemblageDistance(samples_A,samples_B)
        L_max = np.max(distances)
        
        print('............支持向量机加点日志.............')
        print('备选采样点数目:%d'%samples_A.shape[0])
        print('样本集B采样点数目:%d'%samples_B.shape[0])
        print('最大距离:%.4f'%L_max)

        for i in range(sampleCMaximum):
            pos = np.where(distances==L_max)[0]
            if samples_C is None:
                samples_C = samples_A[pos[0],:].reshape((1,-1))
            else:
                samples_C = np.vstack((samples_C,samples_A[pos[0],:]))
                if samples_C.shape[0]>sampleCMaximum:
                    break
            samples_B = np.vstack((samples_B,samples_A[pos[0],:]))
            samples_A = np.delete(samples_A,pos,axis=0)
            distances = self.AssemblageDistance(samples_A,samples_B)
            L_max = np.max(distances)

        if samples_C is None:
            print('sample_C集合为空，分割超平面两侧点密度达到要求')
            return samples_C
        else:
            print('加点数目:%d'%samples_C.shape[0])
            print('加点之后最大距离:%.4f'%L_max)

        if self.f.dim == 2:
            plt.scatter(samples_A[:,0],samples_A[:,1],c='r',marker='.')
            plt.scatter(samples_B[:,0],samples_B[:,1],c='b',marker='.') 
            plt.scatter(samples_C[:,0],samples_C[:,1],c='c',marker='.')                        
            plt.xlim(self.f.l[0]-0.1,self.f.u[0]+0.1)
            plt.ylim(self.f.l[1]-0.1,self.f.u[1]+0.1)
            import time
            timemark = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            path = self.logPath+'/SVM_Photo_{0}.png'.format(timemark)
            plt.savefig(path)
            plt.show(5)

        return samples_C      

    def infillSample1(self,svm,samples,T0,T1,labelNum):
        '''超平面边界加点算法，选取距离超平面最近的数目为candidateNum的样本点，同时用加入样本集C的数目来限制加点密度\n
        in : \n
        svm : 支持向量机实例\n
        samples : 已计算的样本点\n
        T0 : 初始样本集门限\n
        T1 : 入选sampleC门限\n
        labelNum : 一维向量，维度与fit函数的x的shape[1]相同，表示产生初始候选集时各维度的划分数目\n
        out : \n
        samples_C : 二维矩阵，每一行是一个样本点。若为None，代表超平面两侧采样点密度满足要求\n
        '''
        #检查参数维度是否匹配
        dim = self.f.dim
        if dim!=len(labelNum):
            raise ValueError('infillSample:参数维度不匹配')


        #生成样本集A，B，C
        samples_A = self.infillSpace(labelNum)
        samples_B = samples
        samples_C = None


        dis_A = svm.decision_function(samples_A)
        samples_A = samples_A[np.where(np.abs(dis_A)<=T0)] 
        
        #对于样本集B门限约束固定为1.1
        T0_B = 1.1
        dis_B = svm.decision_function(samples_B)
        samples_B = samples_B[np.where(np.abs(dis_B)<T0_B)] 

        if samples_B.shape[0] == 0:
            raise ValueError('infillSample:T0设置过小，区域内没有已采样点')

        #计算样本集A与样本集B的距离
        distances = self.AssemblageDistance(samples_A,samples_B)
        L_max = np.max(distances)
        
        print('............支持向量机加点日志.............')
        print('备选采样点数目:%d'%samples_A.shape[0])
        print('样本集B采样点数目:%d'%samples_B.shape[0])
        print('最大距离:%.4f'%L_max)

        while L_max>T1:
            pos = np.where(distances==L_max)[0]
            if samples_C is None:
                samples_C = samples_A[pos[0],:].reshape((1,-1))
            else:
                samples_C = np.vstack((samples_C,samples_A[pos[0],:]))
            samples_B = np.vstack((samples_B,samples_A[pos[0],:]))
            samples_A = np.delete(samples_A,pos,axis=0)
            distances = self.AssemblageDistance(samples_A,samples_B)
            L_max = np.max(distances)

        if samples_C is None:
            print('sample_C集合为空，分割超平面两侧点密度达到要求')
            return samples_C
        else:
            print('加点数目:%d'%samples_C.shape[0])
            print('加点之后最大距离:%.4f'%L_max)

        if self.f.dim == 2:
            plt.scatter(samples_A[:,0],samples_A[:,1],c='r',marker='.')
            plt.scatter(samples_B[:,0],samples_B[:,1],c='b',marker='.') 
            plt.scatter(samples_C[:,0],samples_C[:,1],c='c',marker='.')                        
            plt.xlim(self.f.l[0]-0.1,self.f.u[0]+0.1)
            plt.ylim(self.f.l[1]-0.1,self.f.u[1]+0.1)
            import time
            timemark = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            path = self.logPath+'/SVM_Photo_{0}.png'.format(timemark)
            plt.savefig(path)
            plt.show(5)

        return samples_C      

    def infillSample2(self,svm,samples,T0,sampleCMaximum):
        '''超平面边界加点算法，选取距离超平面T0之内的样本点，同时用加入样本集C的数目来限制加点密度\n
        in : \n
        svm : 支持向量机实例\n
        samples : 已计算的样本点\n
        T0 : 初始样本集门限\n
        sampleCMaximum : sample_C的最大数目，如果超过此数，加点终止\n
        labelNum : 一维向量，维度与fit函数的x的shape[1]相同，表示产生初始候选集时各维度的划分数目\n
        out : \n
        samples_C : 二维矩阵，每一行是一个样本点。若为None，代表超平面两侧采样点密度满足要求\n
        '''

        #生成样本集A，B，C
        samples_A = self.f.data[:,:self.f.dim]
        samples_B = samples
        samples_C = None


        dis_A = svm.decision_function(samples_A)
        samples_A = samples_A[np.where(np.abs(dis_A)<=T0)] 
        if samples_A.shape[0] < sampleCMaximum:
            return samples_A      
        
        #对于样本集B门限约束固定为1.1
        T0_B = 1.1
        dis_B = svm.decision_function(samples_B)
        samples_B = samples_B[np.where(np.abs(dis_B)<T0_B)] 

        if samples_B.shape[0] == 0:
            raise ValueError('infillSample:T0设置过小，区域内没有已采样点')

        #计算样本集A与样本集B的距离
        distances = self.AssemblageDistance(samples_A,samples_B)
        L_max = np.max(distances)
        
        print('............支持向量机加点日志.............')
        print('备选采样点数目:%d'%samples_A.shape[0])
        print('样本集B采样点数目:%d'%samples_B.shape[0])
        print('最大距离:%.4f'%L_max)

        for i in range(sampleCMaximum):
            pos = np.where(distances==L_max)[0]
            if samples_C is None:
                samples_C = samples_A[pos[0],:].reshape((1,-1))
            else:
                samples_C = np.vstack((samples_C,samples_A[pos[0],:]))
                if samples_C.shape[0]>sampleCMaximum:
                    break
            samples_B = np.vstack((samples_B,samples_A[pos[0],:]))
            samples_A = np.delete(samples_A,pos[0],axis=0)
            distances = self.AssemblageDistance(samples_A,samples_B)
            L_max = np.max(distances)

        if samples_C is None:
            print('sample_C集合为空，分割超平面两侧点密度达到要求')
            return samples_C
        else:
            print('加点数目:%d'%samples_C.shape[0])
            print('加点之后最大距离:%.4f'%L_max)

        return samples_C  

    def infillSample3(self,svm,samples,T0,sampleCMaximum):
        '''针对高维问题的超平面边界加点算法，选取距离超平面T0之内的样本点，同时用加入样本集C的数目来限制加点密度\n
        in : \n
        svm : 支持向量机实例\n
        samples : 已计算的样本点\n
        T0 : 初始样本集门限\n
        sampleCMaximum : sample_C的最大数目，如果超过此数，加点终止\n
        labelNum : 一维向量，维度与fit函数的x的shape[1]相同，表示产生初始候选集时各维度的划分数目\n
        out : \n
        samples_C : 二维矩阵，每一行是一个样本点。若为None，代表超平面两侧采样点密度满足要求\n
        '''
        #生成样本集A，B，C

        samples_B = samples
        samples_C = None

        samples_A = np.array([])
        bins = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2])

        while samples_A.shape[0]<sampleCMaximum*10:
            mc = PseudoMonteCarlo(bins,10,self.f.l,self.f.u)
            points = mc.realSamples
            dis_A = svm.decision_function(points)
            points = points[np.where(np.abs(dis_A)<=T0)]
            if points.shape[0]==0:
                continue
            else:
                if samples_A.shape[0]==0:
                    samples_A = points
                else:
                    samples_A = np.vstack((samples_A, points))
        
        #对于样本集B门限约束固定为1.1
        T0_B = 1.1
        dis_B = svm.decision_function(samples_B)
        samples_B = samples_B[np.where(np.abs(dis_B)<T0_B)] 

        if samples_B.shape[0] == 0:
            raise ValueError('infillSample:T0设置过小，区域内没有已采样点')

        #计算样本集A与样本集B的距离
        distances = self.AssemblageDistance(samples_A,samples_B)
        L_max = np.max(distances)
        
        print('............支持向量机加点日志.............')
        print('备选采样点数目:%d'%samples_A.shape[0])
        print('样本集B采样点数目:%d'%samples_B.shape[0])
        print('最大距离:%.4f'%L_max)

        for i in range(sampleCMaximum):
            pos = np.where(distances==L_max)[0]
            if samples_C is None:
                samples_C = samples_A[pos[0],:].reshape((1,-1))
            else:
                samples_C = np.vstack((samples_C,samples_A[pos[0],:]))
                if samples_C.shape[0]>sampleCMaximum:
                    break
            samples_B = np.vstack((samples_B,samples_A[pos[0],:]))
            samples_A = np.delete(samples_A,pos,axis=0)
            distances = self.AssemblageDistance(samples_A,samples_B)
            L_max = np.max(distances)

        if samples_C is None:
            print('sample_C集合为空，分割超平面两侧点密度达到要求')
            return samples_C
        else:
            print('加点数目:%d'%samples_C.shape[0])
            print('加点之后最大距离:%.4f'%L_max)

        if self.f.dim == 2:
            plt.scatter(samples_A[:,0],samples_A[:,1],c='r',marker='.')
            plt.scatter(samples_B[:,0],samples_B[:,1],c='b',marker='.') 
            plt.scatter(samples_C[:,0],samples_C[:,1],c='c',marker='.')                        
            plt.xlim(self.f.l[0]-0.1,self.f.u[0]+0.1)
            plt.ylim(self.f.l[1]-0.1,self.f.u[1]+0.1)
            import time
            timemark = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            path = self.logPath+'/SVM_Photo_{0}.png'.format(timemark)
            plt.savefig(path)
            plt.show(5)

        return samples_C   

    def Test_SVM_Kernal(self):
        #理论分割函数
        f = self.f   

        data = np.loadtxt(self.logPath+'/A_Samples.txt')
        samples = data[:,0:f.dim]
        mark = data[:,f.dim+1]

        # for i in range(0,10):
        #     if i == 0:
        #         d = 1
        #     else:
        #         d = i*0.1
        #     # print('线性核函数,指数:%.4f'%d)
        #     print('线性核函数,惩罚系数:%.4f'%d)            
        #     svm=SVM_SKLearn.SVC(C=d,kernel='linear')
        #     svm.fit(samples,mark)
        #     # print(svm.decision_function(samples))
        #     f.report(svm)

        # 用同样的方法检测高斯核函数
        # for i in range(1,10):
        #     g = i*0.00001
        #     print('\n高斯核函数,指数:%.6f'%g)
        #     svm=SVM_SKLearn.SVC(C=1000,kernel='rbf',gamma=g)            
        #     # print('\n高斯核函数,惩罚系数:%.6f'%g)            
        #     # svm=SVM_SKLearn.SVC(C=g,kernel='rbf',gamma=0.00008) 
        #     svm.fit(samples,mark) 
        #     f.report(svm)           
        # for i in range(1,10):
        #     g = i*0.0001
        #     print('\n高斯核函数,指数:%.6f'%g)
        #     svm=SVM_SKLearn.SVC(C=1000,kernel='rbf',gamma=g)            
        #     # print('\n高斯核函数,惩罚系数:%.6f'%g)            
        #     # svm=SVM_SKLearn.SVC(C=g,kernel='rbf',gamma=0.00008) 
        #     svm.fit(samples,mark) 
        #     f.report(svm)   

        for i in range(1,10):
            g = i*1000
            # print('\n高斯核函数,指数:%.6f'%g)
            # svm=SVM_SKLearn.SVC(C=1000,kernel='rbf',gamma=g)            
            print('\n高斯核函数,惩罚系数:%.6f'%g)            
            svm=SVM_SKLearn.SVC(C=g,kernel='rbf',gamma=0.0001) 
            svm.fit(samples,mark) 
            f.report(svm)       


def SKCO_test():
    f = TestFunction_G4_Simple()
    # f.TestData()

    skco = SKCO(f,'./Data/G4简化函数测试1')
    # skco.Step_A(31,10)
    # skco.Test_SVM_Kernal()    

    # skco.Step_B([0.1,0.1,0.1,0.1,0.1],[10,10,10,10,10]) 
    skco.Step_C()



if __name__=='__main__':
    SKCO_test()

