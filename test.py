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
    a = np.array([[2,3],[4,5]])
    a = np.delete(a,1,axis=0)
    print(a)

 




if __name__=='__main__':
    test24()





