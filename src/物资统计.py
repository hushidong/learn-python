# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#step 1: io
#目的是对excel文件进行输入输出
#读取excel文件
df=pd.read_excel("D:/w_cpython/dataanalysis/book.xlsx",'Sheet1')
print(df)
#写入excel文件
print(df.to_excel('D:/w_cpython/dataanalysis/foo.xlsx', sheet_name='Sheet1'))
print()

#step 2: sort
#目的是根据列的信息进行排序
# a：确定行和列的标签
print(df.index)
print(df.columns)
# b：根据书目列进行排序
# print(df.sort_index(by='书目')) #这种用法过时了，给出了警告
#print(df.sort_values(by='书目')) #或采用下面一句
df1=df.sort_values(by=df.columns[0])
print(df1) #这样也相同
print(df1.to_excel('D:/w_cpython/dataanalysis/booksorted.xlsx', sheet_name='Sheet1'))
print()


#step 3：statistic
#目的是统计书的总数，价格总数等
#print(df1.describe())
#print()
#print(df1.sum().to_frame().T)
#print()
print(df1[['总价','数量']].sum().to_frame().T)
print()
ps=df1['总价'].sum()
ns=df1['数量'].sum()
print('总金额=',ps)
print('总数量=',ns)
print('平均单价=',df1['单价'].mean())
print('书的平均价格=',ps/ns)
print()

#step 4：statistic
#目的是统计有哪几种书，几种书的数量是多少
shumu=df1['书目'].unique() #确定所有的不同书目
print(type(shumu))
print('书目有:')
print(shumu)
print()

nshumu=df1['数量'].groupby(df1['书目']).sum()
print(type(nshumu))
nsm=nshumu.append(pd.Series([0],index=['小计']))#添加一个值为0，索引为小计的series
nsm[-1]=nsm.sum() #series,也可以使用iloc来定位
print(nsm)
print(nsm.to_excel('D:/w_cpython/dataanalysis/书目数量.xlsx', sheet_name='Sheet1'))
print()

zongjia=df1[['总价','数量']].groupby(df1['书目']).sum()
zjb=zongjia.append(pd.DataFrame([[0,0]],index=['小计'],columns=zongjia.columns))#添加一个行为[0,0]，索引为小计的dataframe
zjb.iloc[-1,0]=zjb[zjb.columns[0]].sum() #需要使用iloc来定位
zjb.iloc[-1,1]=zjb[zjb.columns[1]].sum()
print(zjb)
print(zjb.to_excel('D:/w_cpython/dataanalysis/书目总价.xlsx', sheet_name='Sheet1'))
print()

#step 5：statistic
#目的是统计出所有相同书的信息
df2=df1.set_index(['书目'],drop=False) #重新做索引，方便后面处理
#print(df2)
lstbooks=list(shumu)
#print(lstbooks)
for booktitle in lstbooks: #对所有书目做遍历
    lstidx=list(df2.columns)
    #print(lstidx)
    df4=df2.loc[booktitle]
    #print(df4)
    if(isinstance(df4,pd.core.series.Series)):#当只有一本书时，pandas自动转为series，要特殊处理
        df5=df4.to_frame().T
        sr1=pd.Series(['小计',0,0,0,0],index=lstidx)
        df6=df5.append(sr1,ignore_index=True)#在dataframe中加入一个series
    else:
        sr1=pd.Series(['小计',0,0,0,0],index=lstidx)
        df6=df4.append(sr1,ignore_index=True)#ignore_index=True,dataframe中加series必须要设该选项
    df6.iloc[-1,2]=df6['数量'].sum()
    df6.iloc[-1,3]=df6['总价'].sum()
    print(df6)
    print(df6.to_excel('D:/w_cpython/dataanalysis/'+booktitle+'.xlsx', sheet_name='Sheet1'))
    print()

#step 6：statistic
#目的是统计出价格差超过10的书籍名
print('各书目的单价最大差值：')
jiacha=df1['单价'].groupby(df1['书目']).max()-df1['单价'].groupby(df1['书目']).min()
print(jiacha)
print()
print('单价差超过10的书目：')
jiachalist=jiacha[jiacha>=10]#这里有一个条件筛选
print(jiachalist)
print()
lstbookjiacha=list(jiachalist.index)
print('价差超过10的书目:',lstbookjiacha)
print()

#统计出价格差超过10的书籍的信息
for booktitle in lstbookjiacha:
    lstidx=list(df2.columns)
    df3=df2.loc[booktitle]
    #print(df3)
    sr1=pd.Series(['小计',0,0,0,0],index=lstidx)
    df7=df3.append(sr1,ignore_index=True)#ignore_index=True,dataframe中加series必须要设该选项
    df7.iloc[-1,2]=df7['数量'].sum()
    df7.iloc[-1,3]=df7['总价'].sum()
    print(df7)
    print()


























