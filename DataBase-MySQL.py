# ***************************************************************************
# Copyright (c) 2018 西安交通大学
# All rights reserved
# 
# 文件名称：DataBase-MySQL.py
# 
# 摘    要：与MySQL数据库交互数据。由于目前MySQL只支持python2，所以程序只可以运行在python2的环境下。
#           普通数据存储于文件中，如果是重要的数据样本，使用本程序将数据迁移到数据库中。
# 
# 创 建 者：上官栋栋
# 
# 创建日期：2018年11月13日
#
# 修改记录
# 日期  修改者   		版本     修改内容
# ------------- 		-------  ------------------------  
# ***************************************************************************

#coding=utf-8

import sys
import MySQLdb

print "输入数据库root用户密码："
password = sys.stdin.readline().strip()


def test1():
    '''插入数据'''
    #host是ip地址，本地可以使用localhost代替
    conn = MySQLdb.connect(host='localhost',user='root',passwd=password,db='OUC',port=3306,charset='utf8')

    cur = conn.cursor()
    cur.execute("insert into users(username,password,email) values (%s,%s,%s)",("python","123123","python@gmail.com"))
    #将一句指令多次执行
    cur.executemany("insert into users (username,password,email) values (%s,%s,%s)",\
        (("google","111222","g@gmail.com"),\
        ("facebook","222333","f@face.book"),\
        ("github","333444","git@hub.com"),\
        ("docker","444555","doc@ker.com")))
    conn.commit()

    cur.close()
    conn.close()

def test2():
    '''数据查询'''
    conn = MySQLdb.connect(host='localhost',user='root',passwd=password,db='OUC',port=3306,charset='utf8')

    cur = conn.cursor()
    cur.execute("select * from users")
    #获取所有数据
    lines = cur.fetchall()
    for line in lines:
        print line
    
    #由于光标移动重新查询
    cur.execute("select * from users") 

    #获取一条记录 
    print cur.fetchone()
    #光标向下移动一条记录，默认是“relative”
    cur.scroll(1)
    print cur.fetchone()
    cur.scroll(-2,'relative')
    print cur.fetchone()

    #移动到第一条记录
    cur.scroll(0,'absolute')
    print cur.fetchone()

    #从当前位置获取指定数目的记录
    lines = cur.fetchmany(1)
    for line in lines:
        print line


    cur.close()

    cur_dict = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)
    cur_dict.execute("select * from users")
    lines = cur_dict.fetchall()
    for line in lines:
        print line

    cur_dict.close()
    conn.close()
    
def test3():
    '''修改数据'''
    conn = MySQLdb.connect(host='localhost',user='root',passwd=password,db='OUC',port=3306,charset='utf8')

    cur = conn.cursor()

    cur.execute("update users set username=%s where id=2",("mypython",))
    conn.commit()

    cur.execute("select * from users where id=2")
    print cur.fetchone()

    cur.close()
    conn.close()


if __name__=='__main__':
    test3()
