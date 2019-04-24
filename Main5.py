#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***************************************************************************
# Copyright (c) 2019 西安交通大学
# All rights reserved
# 
# 文件名称：Main5.py
# 
# 摘    要：针对Rotor37的实例优化
#
# 创 建 者：上官栋栋
# 
# 创建日期：2019年4月12号
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

class TestFunction_Rotor37(object):
    '''
    Rotor37,实例测试
    '''
    def __init__(self):
        '''建立目标和约束函数'''

        self.dim = 4
        self.l = [0,0,0,0]
        self.u = [1,1,1,1]

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


class SKCO(object):
    '''基于SVM和kriging的含约束优化方法,针对实际设计问题\n
    input :\n
    logPath : 日志文件存储位置，用于存储计算中产生的数据，日志'''
    def __init__(self,logPath,UpperBound,LowerBound):
        '''初始化函数'''
        self.logPath = logPath
        if len(UpperBound)!=len(LowerBound):
            raise ValueError('取值上界与取值下界维度不匹配')
        self.l = LowerBound
        self.u = UpperBound
        self.dim = len(LowerBound)
        import os 
        if not os.path.exists(logPath):
            os.makedirs(logPath) 
        self.data = np.loadtxt(self.logPath+'/设计空间密集样本.txt')

    def GenerateInitSample(self,initSampleNum = 100):
        '''
        生成初始样本集
        '''    

        lh=LatinHypercube(self.dim,initSampleNum,self.l,self.u)
        samples=lh.realSamples

        np.savetxt(self.logPath+'/InitSamples.txt',samples,delimiter='\t',fmt='%.18f')

    def GenerateAuxiliarySample(self,auxiliarySampleNum = 10):
        '''初步搜索设计空间\n
        input : \n
        initSampleNum : 整型，初始采样点数目\n
        auxiliarySampleNum : 整型，附加采样点数目\n'''

        samples = np.loadtxt(self.logPath+'/InitSamples.txt')
        value = np.loadtxt(self.logPath+'/初始样本集计算结果/Polytropic_efficiency.txt')
        #建立响应面
        kriging = Kriging()
        theta = [0.77412787, 1.07789896, 0.12595146,1.46702414]
        kriging.fit(samples, value, self.l, self.u,theta)
        
        # print('正在优化theta参数...')
        # theta = kriging.optimize(10000,self.logPath+'/theta优化种群数据.txt')

        #原则上应该计算新的样本点再赋值,但是过于频繁有些麻烦,所以一次性生成了
        for k in range(auxiliarySampleNum):
            print('第%d次加点...'%(k+1))
            nextSample = kriging.nextPoint_Varience()
            samples = np.vstack([samples,nextSample])
            nextValue = kriging.get_Y(nextSample)+np.random.randint(-1,1)*0.01
            value = np.append(value,nextValue)
            kriging.fit(samples, value, self.l, self.u, theta)

        np.savetxt(self.logPath+'/A_Samples.txt',samples,fmt='%.18f')

    def MoveHypeplane(self,T0,T1):
        '''
        应用SVM分割设计空间，并按照T1_list中的参数设置优化超平面
        '''
        samples = np.loadtxt(self.logPath+'/A_Samples.txt')
        mark = np.loadtxt('./Data/Rotor37实例测试/初始样本集计算结果/Mark.txt')

        svm=SVM_SKLearn.SVC(C=1000,kernel='rbf',gamma=0.006)
        svm.fit(samples,mark)
        self.Report_SVM2(svm)

        new_x = self.infillSample2(svm,samples,T0,T1)
        if new_x is None:
            print('当T1设置为%.2f时，加点数目为0'%T1)
            return
        else:
            print('新加点数目为%d'%new_x.shape[0])
            samples = np.vstack((samples,new_x))


        np.savetxt(self.logPath+'/B_Samples1.txt',samples,fmt='%.18f',delimiter='\t')
        np.savetxt(self.logPath+'/超平面第一次移动加点.txt',new_x,fmt='%.18f',delimiter='\t')
     
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
        dim = self.dim
        if dim != len(labelNum):
            raise ValueError('infillSpace:参数维度不匹配')     

        up = self.u
        low = self.l       

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
        np.savetxt(self.logPath+'/设计空间密集样本.txt',samples,fmt='%.18f',delimiter='\t')

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
        dim = self.dim
        if dim!=len(labelNum):
            raise ValueError('infillSample:参数维度不匹配')


        #生成样本集A，B，C
        samples_A = self.data
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
        samples_A = self.data
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
        '''
        应用贪心算法与交叉验证选择合适的核函数指数与惩罚系数
        '''
        self.samples = np.loadtxt(self.logPath+'/A_Samples.txt')
        self.value = np.loadtxt('./Data/Rotor37实例测试/初始样本集计算结果/Isentropic_efficiency.txt')
        self.mark = np.loadtxt('./Data/Rotor37实例测试/初始样本集计算结果/Mark.txt')
        K = 10
        y = None
        x = None
        for i in range(5):
            for j in range(9):
                # weight = 0.000001*(10**i)
                weight = 0.1*(10**i)
                g = weight*(j+1)
                svm=SVM_SKLearn.SVC(C=g,kernel='rbf',gamma=0.006) 
                print('C=',g,'gamma=',0.006)
                outcome = self.Report_SVM(svm,K)
                if x is None:
                    x = np.array([g])
                    y = outcome
                else:
                    y = np.vstack((y,outcome))
                    x = np.append(x,g)
                print(outcome)
        
        x = np.log10(x)
        fig,(ax0,ax1) = plt.subplots(ncols = 2,figsize=(10,4))
        ax0.plot(x,y[:,0])
        ax0.set_title('Average Accuracy')

        ax1.plot(x,y[:,1])
        ax1.set_title('positive sample judgement')
        plt.show()

    def Report_SVM(self,svm,K):
        '''
        应用交叉验证方法测试SVM模型的性能\n
        input:\n
        svm : 支持向量机实例\n
        K : K折交叉验证法\n
        output : \n
        精度的平均值
        '''

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=K)

        pos_sample = self.samples[np.where(self.mark==1)]
        outcome = None

        for train_index,test_index in skf.split(self.samples,self.mark):

            train_data = self.samples[train_index,:]
            train_mark = self.mark[train_index]
            svm.fit(train_data,train_mark)

            test_data = self.samples[test_index,:]
            test_mark = self.mark[test_index]
            svm_mark = svm.predict(test_data)

            accuracy = np.sum(svm_mark==test_mark)/test_mark.shape[0]
            
            pos_mark = svm.predict(pos_sample)
            pos_num = np.sum(pos_mark==1)

            row = np.array([[accuracy,pos_num]])
            if outcome is None:
                outcome = row
            else:
                outcome = np.vstack((outcome,row))


        outcome = np.mean(outcome,axis=0)           
        return outcome
 
    def Report_SVM2(self,svm):
        '''
        分析SVM的正反例的空间占比
        '''
        mark = svm.predict(self.data)         
        pos_num = np.sum(mark==1)
        neg_num = np.sum(mark==-1)
        total_num = mark.shape[0]

        print('正例数目:%d,占比:%.4f'%(pos_num,pos_num/total_num))
        print('反例数目:%d,占比:%.4f'%(neg_num,neg_num/total_num))





def getDataFromMF1(aimName):
    '''
    读取一个种群的指定名称的数据
    '''
    fileNum = 71
    import re
    reg_str = re.compile(aimName+r'.*')
    reg_float = re.compile(r'-?\d+\.\d*e?-?\d*')
    data = None
    for i in range(fileNum):
        filePath = './Data/Rotor37实例测试/初始样本集计算结果/OptRst/Gen_0/Ind_%d/stage_0.mf'%i
        with open(filePath,'r') as file:
            lines = file.readlines()
            for line in lines:
                match = reg_str.search(line)
                if match:
                    row = reg_float.findall(line)
                    for i in range(len(row)):
                        row[i] = float(row[i])
                    row = np.array(row).reshape((1,-1))
                    if data is None:
                        data = row
                    else:
                        data = np.vstack((data,row))
                    break
    data = np.mean(data,axis=1)
    np.savetxt('./Data/Rotor37实例测试/初始样本集计算结果/%s.txt'%aimName,data,delimiter='\t',fmt='%.18f')    
       
def getDataFromMF2():
    '''
    读取网格无关性文件中数据
    '''
    filePath1 = './Data/Rotor37实例测试/网格无关性验证/NACA_R37_282621/NACA_R37_282621.mf'
    filePath2 = './Data/Rotor37实例测试/网格无关性验证/NACA_R37_301773/NACA_R37_301773.mf'
    filePath3 = './Data/Rotor37实例测试/网格无关性验证/NACA_R37_NACARotor37_1205Experiment-320925/NACA_R37_NACARotor37_1205Experiment-320925.mf'
    filePath4 = './Data/Rotor37实例测试/网格无关性验证/NACA_R37_422613/NACA_R37_422613.mf'
    filePath5 = './Data/Rotor37实例测试/网格无关性验证/NACA_R37_566481/NACA_R37_566481.mf'
    filePath6 = './Data/Rotor37实例测试/网格无关性验证/NACA_R37_705789/NACA_R37_705789.mf'
    filePath7 = './Data/Rotor37实例测试/网格无关性验证/NACA_R37_1040037/NACA_R37_1040037.mf'
    filePathList = [filePath1, filePath2, filePath3, filePath4, filePath5, filePath6, filePath7]
    import re
    aimName = 'Polytropic_efficiency'
    reg_str = re.compile(aimName+r'.*')
    reg_float = re.compile(r'-?\d+\.\d*e?-?\d*')
    data = None
    for filePath in filePathList:
        with open(filePath,'r') as file:
            lines = file.readlines()
            for line in lines:
                match = reg_str.search(line)
                if match:
                    row = reg_float.findall(line)
                    for i in range(len(row)):
                        row[i] = float(row[i])
                    row = np.array(row).reshape((1,-1))
                    if data is None:
                        data = row
                    else:
                        data = np.vstack((data,row))
                    break   
    print(data)

    X = np.array([1,2,3,4,5,6,7])
    plt.plot(X,data)
    plt.show()

def Histogram(aimName):
    '''
    读取约束项,绘制饼状图分析可行域占比
    '''
    
    data = np.loadtxt('./Data/Rotor37实例测试/初始样本集计算结果/%s.txt'%aimName)
    plt.hist(data,10,rwidth=0.5)
    plt.title(aimName)
    plt.savefig('./Data/Rotor37实例测试/初始样本集计算结果/%s_%d.png'%(aimName,data.shape[0]))
    plt.show()
   
def PieChart():

    TPR = np.loadtxt('./Data/Rotor37实例测试/初始样本集计算结果/Absolute_total_pressure_ratio.txt')
    MF = np.loadtxt('./Data/Rotor37实例测试/初始样本集计算结果/Mass_flow.txt')

    TPR_u = 2.05*1.03
    TPR_l = 2.05*(1-0.0135)
    MF_u = 20.75*1.03
    MF_l = 20.75*(1-0.02)
    print('TPR_u',TPR_u)
    print('TPR_l',TPR_l)   
    print('MF_u',MF_u) 
    print('MF_l',MF_l)

    num = TPR.shape[0]

    mark_TRP = np.zeros(num)+1
    mark_MF = np.zeros(num)+2

    mark_TRP[np.where(TPR>TPR_u)] = -1
    mark_TRP[np.where(TPR<TPR_l)] = -1

    mark_MF[np.where(MF>MF_u)] = -2
    mark_MF[np.where(MF<MF_l)] = -2

    mark = mark_MF + mark_TRP
    num1 = np.sum(mark==3)
    num2 = np.sum(mark==-1)
    num3 = np.sum(mark==1)
    num4 = np.sum(mark==-3)
    labels = ['FS','vio-MF','vio-TPR','IFS']
    sizes = [num1,num2,num3,num4]
    explode = (0.1,0.1,0.1,0.1)
    plt.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%')
    plt.show()
    print(labels,sizes)

    
    mark[np.where(mark<3)] = -1
    mark[np.where(mark==3)] = 1
    np.savetxt('./Data/Rotor37实例测试/初始样本集计算结果/Mark.txt',mark,fmt='%.18f')


def SKCO_test():
    skco = SKCO('./Data/Rotor37实例测试',[1,1,1,1],[0,0,0,0])
    # GenerateInitSample(51)
    # skco.GenerateAuxiliarySample(auxiliarySampleNum = 20)

    # skco.Test_SVM_Kernal()    
    # skco.infillSpace([40,40,40,40])
    skco.MoveHypeplane(0.1,10)






if __name__=='__main__':
    SKCO_test()

    # aimNameList = ['Mass_flow','Absolute_total_pressure_ratio','Isentropic_efficiency']
    # for aimName in aimNameList:
    #     getDataFromMF1(aimName)
    #     Histogram(aimName)
    # PieChart()

