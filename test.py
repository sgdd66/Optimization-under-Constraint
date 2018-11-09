# ***************************************************************************
# Copyright (c) 2018 西安交通大学
# All rights reserved
# 
# 文件名称：test.py
# 
# 摘    要：程序编写中对语言特性的测试文件
# 
# 创 建 者：上官栋栋
# 
# 创建日期：2018年11月9日
#
# 修改记录
# 日期  修改者   		版本     修改内容
# ------------- 		-------  ------------------------  
# ***************************************************************************

#coding=utf-8
import sys 
import numpy as np 

def test1():
    '''逐行读取数据'''
    for line in sys.stdin:
        a = line.split()
        print(a)

def test2():
    '''读取一行，并删除头尾的空格'''
    a = sys.stdin.readline().strip()
    print(a)

def test3():
    """从矩阵中抽出一行或者一列不再保持矩阵的维度，而是转变为向量"""
    mat = np.array([[1,2],[3,4]])
    print(mat[0,:])
    print(mat[:,1])

def test4():
    '''numpy数组排序默认是升序'''
    a = np.array([2,4,6,1,9,3])
    b = np.sort(a)
    print(b)

def test5():
    '''numpy的where函数返回元组类型，第一维是目标值所在行索引，第二维是列索引。且每一维都是数组。'''
    a = np.array([2,4,6,1,9,3,1])
    b = np.sort(a)
    index = np.where(a==b[1])
    print(index)
    print(index[0])

    mat = np.array([[1,2],[3,4],[5,1]])
    index = np.where(mat==1)
    print(index)
    print(index[0])



    
    

if __name__=='__main__':
    test5()
    print()



