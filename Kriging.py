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
    def __init__(self,dataPoints,value,min=None,max=None,maxGen=1):
        """points是插值点，n行d列，d是点的维度，n是点的数目
        min和max是拟合空间的范围，用以对数据进行归一化。如果min和max都等于none，说明点数据已经是归一化之后的数据了"""
        num=dataPoints.shape[0]
        d = dataPoints.shape[1]
        self.ys = value.reshape((num, 1))

        if(min is None and max is None):
            self.points=dataPoints
        else:
            self.points=self.uniformization(dataPoints,min,max)
            self.min=min
            self.max=max

        self.p=np.zeros(d)+2
        self.theta=np.zeros(d)
        self.optimize(maxGen)
        # points = self.points
        # num = points.shape[0]
        # d = points.shape[1]
        # self.theta = np.zeros(d)
        # self.p = np.zeros(d)
        # for i in range(0, d):
        #     self.theta[i] = 10
        #     self.p[i] = 2
        # self.theta[0] = 16.9709375
        # self.theta[1] = 16.9709375
        #
        # self.R = np.zeros((num, num))
        # for i in range(0, num):
        #     for j in range(0, num):
        #         self.R[i, j] = self.correlation(points[i, :], points[j, :])
        # self.R_1 = np.linalg.inv(self.R)
        # print(np.dot(self.R_1,self.R))
        # F = np.zeros((num, 1)) + 1
        # R_1 = self.R_1
        # self.ys = value.reshape((num, 1))
        # ys = self.ys
        #
        # beta0 = np.dot(F.T, R_1)
        # denominator = np.dot(beta0, F)
        # numerator = np.dot(beta0, ys)
        # beta0 = numerator / denominator
        # self.beta0 = beta0
        #
        # factor = ys - beta0 * F
        # sigma2 = np.dot(factor.T, R_1)
        # sigma2 = np.dot(sigma2, factor) / num
        # self.sigma2 = sigma2

    def log_likelihood(self,X):

        points = self.points
        num = points.shape[0]
        d = points.shape[1]
        for i in range(d):
            self.theta[i]=X[i]

        self.R = np.zeros((num, num))
        for i in range(0, num):
            for j in range(0, num):
                self.R[i, j] = self.correlation(points[i, :], points[j, :])
        try:
            self.R_1 = np.linalg.inv(self.R)
        except np.linalg.linalg.LinAlgError as error:
            lhL=-10000
            return lhL
        F = np.zeros((num, 1)) + 1
        R_1 = self.R_1
        ys = self.ys

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
            return -1000

        lgL=-num/2*np.log(sigma2**2)-0.5*np.log(det_R)
        if(lgL>0):
            return -1000
        return lgL

    def optimize(self,maxGen):
        # 设置计算相关系数中所使用的theta和p值
        d=self.points.shape[1]
        min=np.zeros(d)
        max=np.zeros(d)+100
        test = ADE.ADE(min, max, 100, 0.5, self.log_likelihood,False)
        ind = test.evolution(maxGen=maxGen)
        self.log_likelihood(ind.x)

    def getY(self,x):
        num=self.points.shape[0]
        for i in range(x.shape[0]):
            x[i]=(x[i]-self.min[i])/(self.max[i]-self.min[i])
        r=np.zeros((num,1))
        for i in range(0,num):
            r[i]=self.correlation(self.points[i,:],x)
        F=np.zeros((num,1))+1
        R_1=self.R_1
        factor=self.ys-self.beta0*F
        y=np.dot(r.T,R_1)
        y=self.beta0+np.dot(y,factor)

        # f1=np.dot(F.T,R_1)
        # f1=(1-np.dot(f1,r))**2
        # f2=np.dot(F.T,R_1)
        # f2=np.dot(f2,F)
        # f1=f1/f2
        # f2=np.dot(r.T,R_1)
        # f2=np.dot(f2,r)
        # self.varience=self.sigma2*(1-f2+f1)
        return y

    def uniformization(self,points,min,max):
        """将各点缩放在【0,1】*n的空间中"""
        d=points.shape[1]
        num=points.shape[0]
        # max=np.zeros(d)
        # min=np.zeros(d)
        # for i in range(0,d):
        #     max[i]=np.max(points[:,i])
        #     min[i]=np.min(points[:,i])
        p=np.zeros(points.shape)
        for i in range(0,num):
            for j in range(0,d):
                p[i,j]=(points[i,j]-min[j])/(max[j]-min[j])
        return p

    def correlation(self,point1,point2):
        """获取相关系数，point1和point2是一维行向量"""
        d=point1.shape[0]
        R=np.zeros(d)
        theta=self.theta
        p=self.p
        for i in range(0,d):
            R[i]=-theta[i]*np.abs(point1[i]-point2[i])**p[i]
        return np.exp(np.sum(R))



if __name__=="__main__":
