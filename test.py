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
# ------------- 	-------  ------------------------  
# ***************************************************************************

#coding=utf-8
import sys 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt




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

def test6():
    '''
    numpy中的nan不是一个数，不能参与比较（<,>,==)，只能用“is”来判断。\n
    inf具有数的性质，可以参与比较，也可以用“is”来判断。
    '''
    a = np.nan
    b = np.inf

    if a is np.nan:
        print('judge nan by "is" ')

    if a==np.nan:
        print('judge nan by "==" ')

    if b is np.inf:
        print('judge inf by "is" ')

    if b == np.inf:
        print('judge inf by "==" ')

    if a > 0:
        print("compare nan with 0")

    if b > 0:
        print("compare inf with 0 ")

    if -1111>-np.inf:
        print('compare -inf with -1111')

def test7():
    '''is not是合法关键字，not is不是'''
    b = 2

    if b is not np.nan:
        print("is not is right")
    # if b not is np.nan:
    #     print("not is is right")

def test8():
    '''pandas读写csv文件'''
    a = np.arange(1,26,1)
    a = a.reshape((5,5))
    
    b = pd.DataFrame(a,columns=['i1','i2','i3','i4','i5'])
    #！！！写入文件

    b.to_csv('./Data/tmp_1.csv')

    # 使用index和 header选项，把它们的值设置为False,可取消默认写入index和header
    b.to_csv('./Data/tmp_2.csv',index =False,header=False)

    # 可以用to_csv()函数的na_rep选项把空字段替换为你需要的值。常用值有NULL、0和NaN
    b.to_csv('./Data/tmp_3.csv',na_rep="空")

    #！！！读取文件

    csvframe = pd.read_csv('./Data/tmp_1.csv')
    print(csvframe, "\n-----*-----")
    csvframe = pd.read_table('./Data/tmp_1.csv',sep=',')
    print(csvframe, "\n-----*-----")
    # 设置header为None，表示文件没有表头，第一行为数据，添加默认表头
    csvframe = pd.read_csv('./Data/tmp_2.csv',header=None) 
    print(csvframe, "\n-----*-----")
    # 指定表头。我们假设文件中有m列数据，设定的names有n个列名。
    # 如果m>n，默认从最后一列（右侧）开始匹配，多余的右侧第一列作为index，其余数据舍弃
    # 如果m==n，正常匹配
    # 如果m<n，默认从第一列（左侧）开始匹配，多余的列名全部赋值Nan
    csvframe = pd.read_csv('./Data/tmp_2.csv',names=['d1','d2','d3','d4','d5','d6']) 
    print(csvframe, "\n-----*-----")
    #等级索引,可以指定以某一列为索引，支持多列索引。
    csvframe = pd.read_csv('./Data/tmp_3.csv',index_col=['i1','i2']) 
    print(csvframe, "\n-----*-----")

def test9():
    '''pandas读txt文件'''
    # 根据正则解析来辨识间隔符号
    txtframe = pd.read_table('./Data/tmp_4.txt',sep=r'\s+') 
    print(txtframe, "\n-----*-----")
    txtframe = pd.read_table('./Data/tmp_4.txt',sep=r'\s+',header=None,engine='python')
    print(txtframe, "\n-----*-----")
    # 使用skiprows选项，可以排除多余的行。把要排除的行的行号放到数组中，赋给该选项即可。
    txtframe = pd.read_table('./Data/tmp_4.txt',sep=r'\s+',skiprows=[3,4])
    txtframe = txtframe.reset_index(drop=True)
    print(txtframe)    

def test10():
    '''读excel文件'''
    import xlrd

    # 设置路径
    path = './Data/1.xlsx'
    # 打开execl
    workbook = xlrd.open_workbook(path)

    # 输出Excel文件中所有sheet的名字
    print(workbook.sheet_names())

    # 根据sheet索引或者名称获取sheet内容
    Data_sheet = workbook.sheets()[0]  # 通过索引获取
    # Data_sheet = workbook.sheet_by_index(0)  # 通过索引获取
    # Data_sheet = workbook.sheet_by_name(u'名称')  # 通过名称获取


    print(Data_sheet.name)  # 获取sheet名称
    rowNum = Data_sheet.nrows  # sheet行数
    colNum = Data_sheet.ncols  # sheet列数

    # 获取所有单元格的内容
    for i in range(rowNum):
        for j in range(colNum):
            print('{0} '.format(Data_sheet.cell_value(i, j)))

    # 获取整行和整列的值（列表）
    rows = Data_sheet.row_values(0)  # 获取第一行内容
    cols = Data_sheet.col_values(1)  # 获取第二列内容
    print (rows)
    print (cols)

    # 获取单元格内容
    cell_A1 = Data_sheet.cell(0, 0).value
    cell_B1 = Data_sheet.row(0)[1].value  # 使用行索引
    cell_A2 = Data_sheet.col(0)[1].value  # 使用列索引
    print(cell_A1, cell_B1, cell_A2)

    # 获取单元格内容的数据类型
    # ctype:0 empty,1 string, 2 number, 3 date, 4 boolean, 5 error
    print('cell(0,0)数据类型:', Data_sheet.cell(0, 0).ctype)


    # 获取单元格内容为日期的数据
    date_value = xlrd.xldate_as_tuple(Data_sheet.cell_value(1,0),workbook.datemode)
    print(type(date_value), date_value)
    print('%d:%d:%d' % (date_value[0:3]))

def test11():
    '''写excel文件'''
    import xlwt

    path = './Data/2.xlsx'
    # 创建工作簿
    workbook = xlwt.Workbook(encoding='utf-8')

    #创建文字格式
    style = xlwt.XFStyle()   # 初始化样式
    font = xlwt.Font()       # 为样式创建字体
    font.name = 'Times New Roman'
    font.bold = True
    font.color_index = 4
    font.height = 220
    style.font = font

    borders = xlwt.Borders()
    borders.left = 6
    borders.right = 6
    borders.top = 6
    borders.bottom = 6
    style.borders = borders

    #!!!如果对一个单元格重复操作，会引发error。所以在打开时加cell_overwrite_ok=True解决
    sheet1 = workbook.add_sheet(u'sheet1',cell_overwrite_ok=True)  
    #表头内容
    header = [u'甲',u'乙',u'丙',u'丁',u'午',u'己',u'庚',u'辛',u'壬',u'癸']

    # 写入表头
    for i in range(0, len(header)):
        sheet1.write(0, i, header[i], style)
    sheet1.write(1,0,u'合计',style)

    #1,2,3,4表示合并区域是从第2行到第3行，从第4列到第5列，0表示要写入的单元格内容，style表示单元格样式    
    sheet1.write_merge(1,2,3,4,0,style)

    # 保存文件
    workbook.save(path)

def test12():
    '''minidom读取xml文档'''
    #coding=utf-8
    import  xml.dom.minidom

    #打开xml文档
    dom = xml.dom.minidom.parse('./Data/tmp_5.xml')

    #得到文档元素对象
    root = dom.documentElement
    print(root.nodeName)
    #节点值，这个属性只对文本结点有效
    print(root.nodeValue)

    print(root.nodeType)
    print(root.ELEMENT_NODE)

    #通过子节点的nodeName查询结点，由于存在同名子节点，所以返回list类型
    item_list = root.getElementsByTagName('item')
    item1 = item_list[0]
    item2 = item_list[1]
    #通过结点的属性名获取属性值
    id1 = item1.getAttribute('id')
    id2 = item2.getAttribute('id')
    print(item1.nodeName,id1)
    print(item2.nodeName,id2)

    item_list = root.getElementsByTagName('caption')
    item1 = item_list[0]
    item2 = item_list[1]
    #获取结点的数据
    print(item1.firstChild.data)
    print(item2.firstChild.data)

def test13():
    '''Python写入xml文件'''
    import xml.dom.minidom

    impl = xml.dom.minidom.getDOMImplementation()
    dom = impl.createDocument(None, 'employees', None)

    root = dom.documentElement  
    employee = dom.createElement('employee')
    employee.setAttribute('id' ,'1')
    root.appendChild(employee)

    #创建element结点，结点名为name
    nameE=dom.createElement('name')
    #创建文本结点，文本结点只有数据，没有<></>包裹
    nameT=dom.createTextNode('linux')
    #文本结点作为element结点的第一个子节点
    nameE.appendChild(nameT)

    employee.appendChild(nameE)

    ageE=dom.createElement('age')
    ageT=dom.createTextNode('30')
    ageE.appendChild(ageT)
    employee.appendChild(ageE)

    f= open('./Data/tmp_6.xml', 'w', encoding='utf-8')
    dom.writexml(f, addindent='  ', newl='\n',encoding='utf-8')
    f.close()      

def test14():
    '''numpy读取txt文件'''

    data = np.loadtxt('./Data/tmp_2.csv',delimiter=',')
    print(data)
    np.savetxt('./Data/tmp_7.txt',data,fmt='%d',delimiter=',',newline='\n')

def test15():
    '''以负数为索引号可以逆向查找'''
    a = [[-1,2],[2,4],[-1,3],[-1,2]]
    print(a[-2:-1])
    print(a[-2])

def test16():
    '''测试list的count函数'''
    a = [[-1,2],[2,4],[-1,3],[-1,2]]
    print(a.count(2))
    a.remove([-1,2])
    print(a)

def test17():
    '''测试np.mgrid函数'''
    min = np.array([-5, 0])
    max = np.array([10, 15])

    x, y = np.mgrid[min[0]:max[0]:10j, min[1]:max[1]:10j]
    print(x)
    print(y)

def test18():
    '''正太分布的累计分布函数与概率密度函数'''
    from scipy.stats import norm
    print(norm.pdf([0,0.1]))
    print(norm.cdf([0,0.1]))
    print(norm.pdf(0))
    print(norm.cdf(0))

def test19():
    '''comb计算组合函数，perm计算排列函数'''
    from scipy.special import comb, perm
    print(comb(4,2))
    print(perm(4,2))

def test20():
    '''测试文件路径，并建立文件路径'''
    # 引入模块
    import os
    path = './Data/测试用文件夹/'
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    # path=path.rstrip("/")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False
 
def test21():
    '''测试gif文件的制作'''
    import imageio
    root_path = '/home/sgdd/Optimization-under-Constraint/Data/Kriging加点模型测试4'
    for g in range(1,11):
        images_predicte = []
        images_Varience = []
        for k in range(20):
            path = root_path+'/%d/Kriging_Predicte_Model_%d.png'%(g,k)
            images_predicte.append(imageio.imread(path))
            path = root_path+'/%d/Kriging_Varience_Model_%d.png'%(g,k) 
            images_Varience.append(imageio.imread(path))

        imageio.mimsave(root_path+'/Predicte_g_%d.gif'%g, images_predicte,duration=1)
        imageio.mimsave(root_path+'/Varience_g_%d.gif'%g, images_Varience,duration=1)

def test22():
    '''求取矩阵的最小最大值'''
    data = np.random.randint(0,10,size=(4,4))
    print(data)
    min_row = np.min(data,1)
    min_col = np.min(data,0)

    print('min_row:')
    print(min_row)
    print('min_col:')
    print(min_col)

    max_row = np.max(data,1)
    max_col = np.max(data,0)
    print('max_row:')
    print(max_row)
    print('max_col:')
    print(max_col)

def test23():
    g1 = lambda x:x[0]**2-x[1]+1
    g2 = lambda x:1-x[0]+(x[1]-4)**2

    x, y = np.mgrid[0:10:100j,0:10:100j]
    mark = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = [x[i,j],y[i,j]]
            if g1(a)<0 and g2(a)<0:
                mark[i,j] = 1
            else:
                mark[i,j] = -1

    x = x[mark==1]
    y = y[mark==1]
    plt.scatter(x,y)
    a = (1.2279713, 4.2453733)
    plt.scatter(a[0],a[1])
    plt.show()
    
    print(min(x),max(x))
    print(min(y),max(y))

def test24():
    '''制作解释样本点线性分布的示例图片'''
    from DOE import PseudoMonteCarlo


    f = lambda x:np.sin(x)
    end = 4*np.pi
    line_x = np.linspace(0,end,100)
    line_y = f(line_x)

    pt_x = np.arange(0,end+1,np.pi)
    pt_y = f(pt_x)

    coeff = np.polyfit(pt_x,pt_y,7)
    polyFunc = np.poly1d(coeff)
    fit_y = polyFunc(line_x)
    
    
    plt.plot(line_x,line_y)
    plt.scatter(pt_x,pt_y)
    plt.plot(line_x,fit_y)
    plt.show()

    pt_x = PseudoMonteCarlo(np.array([4]),1,[0.5*np.pi],[3.5*np.pi]).realSamples.reshape((-1))
    pt_x = np.append(pt_x,[0,end])

    
    pt_y = f(pt_x)

    pt_x = pt_x.reshape((-1,1))
    pt_y = pt_y.reshape((-1,1))

    from Kriging import Kriging

    kriging = Kriging()
    kriging.fit(pt_x,pt_y,min=[0],max=[end])
    fit_y = np.zeros_like(line_y)
    for i in range(line_x.shape[0]):
        fit_y[i] = kriging.get_Y(np.array([line_x[i]]))


    # def func(x,a,b,c):
    #     return a*np.sin(x+b)+c
    # from scipy.optimize import curve_fit
    # popt,pcov = curve_fit(func,pt_x,pt_y)
    # a = popt[0]
    # b = popt[1]
    # c = popt[2]
    # fit_y = func(line_x,a,b,c)

    plt.plot(line_x,line_y)
    plt.scatter(pt_x,pt_y)
    plt.plot(line_x,fit_y)
    plt.show()

def test25():
    '''测试G4函数'''
    from Main import TestFunction_G4

    f = TestFunction_G4()

    x = np.array([78,33,29.996,45,36.7758])
    y = np.array([78,33,30.04292047,45,36.65603025])
    z = np.array([78,33,29.99790876,45,36.76916756])
    print(f.isOK(x))
    print(f.aim(x))
    print(f.isOK(y))
    print(f.aim(y))    
    print(f.isOK(z))
    print(f.aim(z))    


    func = lambda x : f.aim(x)-100000*np.min([f.isOK(x),0])
    from ADE import ADE
    ade = ADE(f.min, f.max, 100, 0.5, func,True)
    opt_ind = ade.evolution(maxGen=100000)
    print(opt_ind.x)

    pointNum = 1
    for i in range(f.dim):
        pointNum *= f.max[i]-f.min[i]+1
    
    value = np.zeros(pointNum)
    mark = np.zeros(pointNum)
    for i in range(pointNum):
        point = f.min.copy()
        index = i
        for j in range(f.dim):
            point[j] += index%(f.max[j]-f.min[j]+1)
            index = index // (f.max[j]-f.min[j]+1)
            if index == 0:
                break
        value[i] = f.aim(point)
        mark[i] = f.isOK(point)

    value = np.reshape(value,(-1,1))
    mark = np.reshape(mark,(-1,1))
    data = np.hstack((value,mark))
    np.savetxt('./Data/G4函数测试/data.txt',data)  

    posNum = np.sum(mark==1)
    negNum = np.sum(mark == -1)  
    print(posNum,negNum)

def test26():
    '''检查连通域'''
    from Main import TestFunction_G4

    f = TestFunction_G4()    
    weight = np.array(f.max)-np.array(f.min)+1
    
    data = np.loadtxt('./Data/G4函数测试/data.txt')
    pos_num = np.sum(data==1)
    neg_num = np.sum(data==-1)
    total_num = data.shape[0]
    print('positive point num:',pos_num)
    print('positive point ratio:',pos_num/total_num)

    print('negative point num:',neg_num)
    print('negative point ratio:',neg_num/total_num)  

    print(neg_num+pos_num,total_num)

    scaleIndex = 2

    num = data.shape[0]
    for i in range(num):
        if data[i,1] == 1:
            ModifyNeighbour(i,scaleIndex,weight,data)
            scaleIndex += 1

    np.savetxt('./Data/G4函数测试/data1.txt',data)
    mark = data[:,1]
    biggestIndex = np.max(mark)
    for i in range(2,int(biggestIndex)+1):
        print(i,':',np.sum(mark==i))


    # data = np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,1]])
    # data = data.reshape((-1,1))
    # mark = np.zeros_like(data)+1
    # mark[np.where(data==0)] = -1
    # data = np.hstack((data,mark))
    
    # num = data.shape[0]
    # scaleIndex = 2
    # for i in range(num):
    #     if data[i,1] == 1:
    #         ModifyNeighbour(i,scaleIndex,[4,4],data)
    #         scaleIndex += 1
        
    # print(data)

def ModifyNeighbour(index,mark,weight,data):
    '''以点index为起始点，将其所在的连通域内所有点标记为mark\n
    index : 起始点索引\n
    mark : 标记符号\n
    weight : 各维度划分数目，例如[2,3]表示第一维有2个划分，第二维有三个划分，共产生6个采样点\n
    data : 数据，N维两列，N为样本数目。第一列是数据值，第二列是标记，-1表示违背约束，1表示满足约束，2,3,4...是相应的连通域标记\n
    '''
    feature = data[index,1]
    data[index,1] = mark
    dim = len(weight)

    import queue
    q = queue.Queue()
    q.put(index)

    while not q.empty():

        index = q.get()
        pos = Index2Pos(index,weight)

        for i in range(dim):
            if pos[i] != 0:
                neighbour = pos.copy()
                neighbour[i] -= 1
                neighbour = Pos2Index(neighbour,weight)
                if data[neighbour,1]==feature:
                    q.put(neighbour)
                    data[neighbour,1] = mark
            if pos[i] != weight[i]-1:
                neighbour = pos.copy()
                neighbour[i] += 1
                neighbour = Pos2Index(neighbour,weight)
                if data[neighbour,1]==feature:
                    q.put(neighbour)
                    data[neighbour,1] = mark                               

def Pos2Index(pos,weight):
    '''将坐标位置转换为索引号\n
    input :\n
    pos : 一维向量，坐标\n
    weight : 维度权重\n

    output :\n
    index : 索引号'''

    index = 0 
    dim = len(pos)
    for i in range(dim):
        value = 1
        for j in range(i):
            value *= weight[j]
        index += value*pos[i]
    
    return index

def Index2Pos(index,weight):
    '''将索引号转换为坐标位置\n
    input :\n
    index : 索引号\n
    weight : 维度权重\n

    output :\n
    pos : 一维向量，坐标'''

    pos = np.zeros_like(weight)
    dim = len(weight)

    for i in range(dim):
        pos[i]=index%weight[i]
        index = index//weight[i]
        if index == 0:
            break
    
    return pos

def test27():
    '''Python的copy机制'''
    l = [1,[3,4]]
    l1 = l
    l2 = l.copy()
    l1[0] = 0
    print(l1,l)
    l2[0] = 4
    print(l2,l)
    l2[1] = 0 
    print(l2,l)

    a = np.array([2,3,4])
    b = np.append(a,2)
    print(b)

def test28(svm=None):
    '''比较SVM与实际分类差异'''
    if svm is None:
        logPath = './Data/G4函数测试2'   
        from SVM import SVM,Kernal_Gaussian,Kernal_Polynomial
        Kernal_Poly = lambda x,y:(np.dot(x,y)+1)**7
        Kernal_Gau = lambda x,y:np.exp((-np.linalg.norm(x-y)**2)/60)
        svm=SVM(10,kernal=Kernal_Gau,path = logPath)
        print('训练初始支持向量机...')
        svm.retrain(logPath+'/SVM_Step_B.txt',0)  

    from Main import TestFunction_G4
    f = TestFunction_G4()

    pointNum = 1
    dimNum = [12,6,9,9,9]
    weight = np.zeros(f.dim)
    for i in range(f.dim):
        pointNum *= dimNum[i]
        weight[i] = (f.max[i]-f.min[i])/(dimNum[i]-1)   

    # points = np.zeros((pointNum,f.dim))    
    # points_mark = np.zeros(pointNum) 
    # for i in range(f.dim):
    #     points[:,i] = f.min[i]

    # for i in range(pointNum):
    #     index = i
    #     for j in range(f.dim):
    #         points[i,j] += index%dimNum[j]*weight[j]
    #         index = index // dimNum[j]
    #         if index == 0:
    #             break       
    #     points_mark[i] = f.isOK(points[i,:])
    
    # points_mark = points_mark.reshape((-1,1))
    # data = np.hstack((points,points_mark))
    # np.savetxt('./Data/G4函数测试1/G4函数空间.txt',data)

    data = np.loadtxt('./Data/G4函数测试1/G4函数空间.txt')
    points_mark = data[:,f.dim]
    points = data[:,0:f.dim]


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

    x = np.array([78,33,29.996,45,36.7758])
    print('实际最优值坐标：',x)
    print('实际最优值:%.6f'%f.aim(x))
    print('SVM判定:%.4f'%svm.transform(x))

def test29():
    '''读取迭代次数对精度的影响中的数据'''
    import re
    path = './Data/G4函数测试1/迭代次数对SVM的影响and多项式p=7'
    # path = './Data/G4函数测试1/迭代次数对SVM的影响and多项式p=9.txt'
    with open(path,'r') as file:
        texts = file.readlines()
        pos = 0
        data = None
        while pos<len(texts):
            if re.search(r'迭代次数：(\d+)',texts[pos]):
                row = np.zeros(5)
                row[0] = float(re.search(r'迭代次数：(\d+)',texts[pos]).group(1))
                row[1] = float(re.search(r'精度:(.*)',texts[pos+9]).group(1))
                row[2] = float(re.search(r'查准率:(.*)',texts[pos+10]).group(1))
                row[3] = float(re.search(r'查全率:(.*)',texts[pos+11]).group(1))
                row[4] = float(re.search(r'F1:(.*)',texts[pos+12]).group(1))      
                if data is None:
                    data = row          
                else:
                    data = np.vstack((data,row))
                pos = pos+13
            else:
                pos += 1
        np.savetxt('./Data/G4函数测试1/迭代次数对精度的影响数据p=7.txt',data)
        plt.plot(np.log10(data[:,0]),data[:,1],'r')
        plt.plot(np.log10(data[:,0]),data[:,2],'b')
        plt.plot(np.log10(data[:,0]),data[:,3],'g')
        plt.plot(np.log10(data[:,0]),data[:,4],'y')
        plt.legend(['acc','P','R','F1'])
        plt.savefig('./Data/G4函数测试1/迭代次数对SVM的影响and多项式p=7.png')        
        plt.show()

def test30():
    x = np.array([78,33,29.996,45,36.7758])
    from Main import TestFunction_G4
    f = TestFunction_G4()
    #各轮搜索最优点
    data = np.loadtxt('./Data/G4函数测试7/局部样本点.txt')
    points = data[:,:5]
    mark = data[:,6]
    dis = np.linalg.norm(points-x,axis=1)
    dis = dis.reshape(-1,1)
    mark = mark.reshape(-1,1)
    data = np.hstack((dis,mark))
    # print(data[np.where(data[:,0]<10)])
    # print(points[np.where(data[:,0]<10)])
    p = points[np.where(data[:,0]<10)]
    for i in range(p.shape[0]):
        print(f.aim(p[i,:]))

    
    x1 = [78,33,31.80947635,45,32.82840826]
    print(f.aim(x1))
    print(f.aim(x))
    print((f.aim(x1)-f.aim(x))/f.aim(x))
    # print(np.linalg.norm(x1-x))
    # print((f.aim(x1)-f.aim(x))/f.aim(x))

def test31():
    x1 = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,50,50,50,0.5]])
    x2 = np.array([[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,40,40,40,0.4]])
    x3 = np.array([[1,1,1,1,1,1,1,1,1,3,3,3,1]])
    x4 = np.array([[1,2,1,1,1,2,1,1,1,3,3,3,1]])
    x5 = np.array([[1,1,1,2,1,1,1,-1,1,3,3,3,1]])    
    x = np.vstack((x1,x2,x3,x4,x5))
    aim = lambda x:5*(np.sum(x[0:4]))-5*(np.sum(x[0:4]**2))-np.sum(x[4:])
    print(aim(x[2]))

    def aim_matrix(M):
        sum1 = 5*np.sum(M[:,0:4],axis=1)
        sum2 = 5*np.sum(np.power(M[:,0:4],2),axis=1)
        sum3 = np.sum(M[:,4:],axis=1)
        return sum1-sum2-sum3

    print(aim_matrix(x))


    g1 = lambda x:2*x[0]+2*x[1]+x[9]+x[10]-10
    g2 = lambda x:2*x[0]+2*x[2]+x[9]+x[11]-10
    g3 = lambda x:2*x[1]+2*x[2]+x[10]+x[11]-10
    g4 = lambda x:-8*x[0]+x[9]
    g5 = lambda x:-8*x[1]+x[10]
    g6 = lambda x:-8*x[2]+x[11]
    g7 = lambda x:-2*x[3]-x[4]+x[9]
    g8 = lambda x:-2*x[5]-x[6]+x[10]
    g9 = lambda x:-2*x[7]-x[8]+x[11]
    l = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    u = [1,1,1,1,1,1,1,1,1,100,100,100,1]

    constrain = [g1,g2,g3,g4,g5,g6,g7,g8,g9]

    dim = 13

    def isOK(x):

        if len(x) != dim:
            raise ValueError('isOK：参数维度与测试函数维度不匹配')
        
        for i in range(dim):
            if x[i] < l[i] or x[i] > u[i]:
                raise ValueError('isOK: 参数已超出搜索空间')

        for g in constrain:
            if g(x)>0:
                return -1
        return 1

    for i in range(x.shape[0]):
        try:
            print(isOK(x[i]))
        except ValueError:
            print('error')


    def isOK_Matrix(x):
        mark = np.zeros(x.shape[0])+1

        for i in range(dim):
            mark[np.where(x[:,i]<l[i])] = -1
            mark[np.where(x[:,i]>u[i])] = -1

        g1 = 2*x[:,0]+2*x[:,1]+x[:,9]+x[:,10]-10
        g2 = 2*x[:,0]+2*x[:,2]+x[:,9]+x[:,11]-10
        g3 = 2*x[:,1]+2*x[:,2]+x[:,10]+x[:,11]-10
        g4 = -8*x[:,0]+x[:,9]
        g5 = -8*x[:,1]+x[:,10]
        g6 = -8*x[:,2]+x[:,11]
        g7 = -2*x[:,3]-x[:,4]+x[:,9]
        g8 = -2*x[:,5]-x[:,6]+x[:,10]
        g9 = -2*x[:,7]-x[:,8]+x[:,11]
        mark[np.where(g1>0)] = -1
        mark[np.where(g2>0)] = -1        
        mark[np.where(g3>0)] = -1
        mark[np.where(g4>0)] = -1
        mark[np.where(g5>0)] = -1
        mark[np.where(g6>0)] = -1
        mark[np.where(g7>0)] = -1        
        mark[np.where(g8>0)] = -1
        mark[np.where(g9>0)] = -1
            
        return mark
    print(isOK_Matrix(x))

def test32():
    points_mark = np.array([1,1,1,1,3,3,3])
    svm_mark    = np.array([1,3,1,3,1,3,1])
    points_pos = points_mark==1
    points_neg = ~points_pos

    svm_pos = svm_mark==1
    svm_neg = ~svm_pos


    TP = np.sum(points_pos & svm_pos)
    FP = np.sum(svm_pos & points_neg)
    TN = np.sum(points_neg & svm_neg)
    FN = np.sum(svm_neg & points_pos)

    print(points_mark>svm_mark)

    a = np.vstack((points_mark,svm_mark))
    print(np.sum(a==points_mark,axis=1))

    print(TP,FP,TN,FN)

def test33():
    from Main import TestFunction_G8
    from Kriging import writeFile
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
            if s[i,j]>0:
                s[i,j]=np.log10(s[i,j]+1)
            elif s[i,j]<0:
                s[i,j]=-np.log10(-s[i,j]+1)

    path = './Data/约束优化算法测试1/G8_Function.txt'
    writeFile([x,y,s],[],path)

def test34():
    '''
    提取文本中的数据
    '''
    import re
    reg = re.compile(r'最优值\[\[.*\]\]，最优点\[.*\]')
    reg_float = re.compile(r'-?\d+\.\d*e?-?\d*')
    with open('./Data/G4简化函数测试1/局部搜索日志0','r') as file:
        data = None
        texts = file.readlines()
        for text in texts:
            match = reg.search(text)
            if match:
                row = reg_float.findall(match.group(0))
                for i in range(len(row)):
                    row[i] = float(row[i])
                if data is None:                   
                    data = np.array(row)
                else:
                    row = np.array(row)                    
                    data = np.vstack((data,row))


    EI = data[:,0].reshape((-1,1))
    point = data[:,1:]


    from Main1 import TestFunction_G4
    f = TestFunction_G4()
    value = np.zeros(point.shape[0])
    mark = np.zeros(point.shape[0])
    for i in range(point.shape[0]):
        value[i] = f.aim(point[i,:])
        mark[i] = f.isOK(point[i,:])
    value = value.reshape((-1,1))
    mark = mark.reshape((-1,1))    

    optimumPoint = np.array(f.optimum)
    distance = np.linalg.norm(point-optimumPoint,axis=1)
    distance = distance.reshape((-1,1))
    data = np.hstack((point,value,mark,distance,EI))


    np.savetxt('./Data/G4简化函数测试1/计算日志提取数据.txt',data,delimiter='\t')
    print(data.shape)

def test35():
    '''
    计算文本中的数据
    '''
    data = np.loadtxt('./Data/约束优化算法测试1/stepC_5/待计算数据.txt')
    from Main import TestFunction_G8
    f = TestFunction_G8()
    value = np.zeros(data.shape[0])
    mark = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        value[i] = f.aim(data[i,:])
        mark[i] = f.isOK(data[i,:])
    value = value.reshape((-1,1))
    mark = mark.reshape((-1,1))
    data = np.hstack((data,value,mark))
    np.savetxt('./Data/约束优化算法测试1/stepC_5/计算数据.txt',data,delimiter='\t')

def test36():
    '''
    绘制数据折线图
    '''
    data = np.loadtxt('./Data/G16函数测试/超平面移动数据.txt')
    X = data[0,:]
    Y1 = data[1,:]
    Y2 = data[2,:]
    Y3 = data[3,:]
    Y4 = data[4,:]
    L1, = plt.plot(X,Y1)
    L2, = plt.plot(X,Y2)
    L3, = plt.plot(X,Y3)
    L4, = plt.plot(X,Y4)

    # plt.ylim((0,1))

    plt.legend([L1,L2,L3,L4],['Accuracy','Precision','Recall','F1'],loc='lower right')
    plt.show()

def test37():
    '''
    读取全部样本点数据并计算
    '''
    data = np.loadtxt('Data/G16函数测试/全部样本点.txt')
    point = data[:,0:5]
    value = data[:,5]
    mark = data[:,6]

    from Main3 import TestFunction_G16

    f = TestFunction_G16()

    optimumPoint = np.array(f.optimum)
    distance = np.linalg.norm(point-optimumPoint,axis=1)
    distance = distance.reshape((-1,1))
    value = value.reshape((-1,1))
    mark = mark.reshape((-1,1))    

    data = np.hstack((point,value,mark,distance))
    np.savetxt('./Data/G16函数测试/计算日志提取数据.txt',data,delimiter='\t')
    

if __name__=='__main__':
    test37()





