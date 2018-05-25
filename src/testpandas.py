# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt



'''
#当dataframe无法进行复制时，可以用手动的方法，
#先将其转化为np.ndarray，然后再转回来
lstidx=list(df1['书目'])
lstcol=list(df1.columns)
print(lstidx,lstcol)
array1=np.array(df1)
print(array1)
df2=pd.DataFrame(array1,index=lstidx,columns=lstcol)
print(df2)

#下面的方法可以用来替代dataframe的append
lstidx=list(df4.index)
lstcol=list(df4.columns)
lstidx.append('小计')
print(lstidx)
array1=np.array(df4).tolist()
array1.append(['小计',0,0,0,0])
print(array1)
df5=pd.DataFrame(array1,index=lstidx,columns=lstcol)
'''



'''
#使用传递的值列表序列创建序列, 让pandas创建默认整数索引
s = pd.Series([1,3,5,np.nan,6,8])
print(s)
#结果中包含一列数据和一列标签,用values和index分别进行引用
print(s.values)
#可以按照自己的意愿构建标签
print(s.index)
object=pd.Series([2,5,8,9],index=['a','b','c','d'])
print(object)
#对序列进行运算
print(object[object>5])
#可以把Series看成一个字典，使用in进行判断
print('a' in object)
#值是不能直接被索引到的
print(2 in object)

#name或者index.name可以对数据进行重命名


#唯一值，成员资格等方法
#isnull或者notnull可以用于判断数据中缺失值情况
data=pd.Series(['a','a','b','b','b','c','d','d'])
print(data.unique())
print(data.isin(['b']))
print(pd.value_counts(data.values))
print(pd.value_counts(data.values,sort=False))
data=pd.Series(['a','a','b',np.nan,'b','c',np.nan,'d'])
print(data.isnull())
print(data.dropna())
print(data.ffill())
print(data.fillna(0))
'''


'''
#字典数据转换为dataframe
data={'year':[2000,2001,2002,2003],
'income':[3000,3500,4500,6000]}
data=pd.DataFrame(data)
print(data)

data1=pd.DataFrame(data,columns=['year','income','outcome'],
index=['a','b','c','d'])
print(data1)
print(data1['year'])#两种索引是等价的，都是对列进行索引
print(data1.year)
data1['money']=np.arange(4)#增
del data1['outcome']#删


noteSeries = pd.Series(["C", "D", "E", "F", "G", "A", "B"],
index=[1, 2, 3, 4, 5, 6, 7])
weekdaySeries = pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
index=[1, 2, 3, 4, 5, 6, 7])
df3 = pd.DataFrame([noteSeries, weekdaySeries])
print(df3)
df3.loc["No0"] = pd.Series([1, 2, 3, 4, 5, 6, 7],index=[1, 2, 3, 4, 5, 6, 7])
print(df3)
df3.loc["No1"] = pd.Series([1, 2, 3, 4, 5, 6, 7])
print(df3)
print()
df4=df3.append(pd.Series([1, 2, 3, 4, 5, 6, 7],index=[1, 2, 3, 4, 5, 6, 7]),ignore_index=True)
print(df4)
print()
df4=df3.append(pd.Series([1, 2, 3, 4, 5, 6, 7]),ignore_index=True)
print(df4)

'''

'''
dates = pd.date_range('20130101',periods=6)
print(dates)

#使用传递的numpy数组创建数据帧,并使用日期索引和标记列
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
print(df)

#使用传递的可转换序列的字典对象创建数据帧.
df2 = pd.DataFrame({ 'A' : 1.,
'B' : pd.Timestamp('20130102'),
'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
'D' : np.array([3] * 4,dtype='int32'),
'E' : pd.Categorical(["test","train","test","train"]),
'F' : 'foo' })

print(df2)
#所有明确类型
print(df2.dtypes)


#查看数据,查看帧顶部和底部行
print(df.head())
print(df.tail(3))

#显示索引,列,和底层numpy数据
print(df.index)
print(df.columns)
print(df.values)

#描述显示数据快速统计摘要
print(df.describe())

#转置数据
print(df.T)

#按轴排序
print(df.sort_index(axis=1, ascending=False))

data=pd.DataFrame(np.arange(10).reshape((2,5)),index=['c','a'],
columns=['one','four','two','three','five'])
print(data)
print(data.sort_index())
print(data.sort_index(axis=1))
print(data.sort_values(by='one'))
print(data.sort_values(by='one',ascending=False))


#pandas数据访问方法, .at, .iat, .loc, .iloc 和 .ix.
#读取,使用[]选择行片断
print(df.A)
print(df['A'])
print(df[0:3])
print(df['20130102':'20130104'])
#显示标签切片, 包含两个端点
print(df.loc['20130102':'20130104',['A','B']])
#降低返回对象维度
print(df.loc['20130102',['A','B']])
#获取标量值
print(df.loc['20130101','A'])
print(df.loc[dates[0],'A'])
df.at[dates[0],'A']

#按位置选择
df.iloc[3]
df.iloc[3:5,0:2]#使用整数片断,效果类似numpy/python
df.iloc[[1,2,4],[0,2]]
df.iloc[1:3,:]
df.iloc[:,1:3]
df.iloc[1,1]
df.iat[1,1]


'''

series1 = pd.Series([1, 2, 3, 4, 5, 6, 7],
index=["C", "D", "E", "F", "G", "A", "B"])

print("series1['E'] = {} \n".format(series1['E']));
print("series1.E = {} \n".format(series1.E));

df1 = pd.DataFrame({"note" : ["C", "D", "E", "F", "G", "A", "B"],
"weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
index=['1', '2', '3', '4', '5', '6', '7'])
print(df1)
#指定一个索引来访问一行的数据
print("df1.loc['2']:\n{}\n".format(df1.loc['2'])) 
#访问某个范围之内的数据
print("series1.loc['E':'A']=\n{}\n".format(series1.loc['E':'A']));
print("df1.iloc[2:4]=\n{}\n".format(df1.iloc[2:4]))
#访问单个的元素值
print("series1.at['E']={}\n".format(series1.at['E']));
print("df1.iloc[4,1]={}\n".format(df1.iloc[4,1]))


'''
#布尔索引
df[df.A > 0]
#where 操作
df[df > 0]
#使用 isin() 筛选：
df2 = df.copy()
df2['E']=['one', 'one','two','three','four','three']
df2[df2['E'].isin(['two','four'])]
print(df2)
df2[-1]=[1,2,3,4,5,6]
print(df2)
#df2.loc[-1,:]=[1,2,3,4,5] #出错，不能加入与前面日期做的索引不同类型的索引
#print(df2)

#赋值
#赋值一个新列，通过索引自动对齐数据
s1 = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130102',periods=6))
df['F'] = s1
#按标签赋值
df.at[dates[0],'A'] = 0
#按位置赋值
df.iat[0,1] = 0
#通过numpy数组分配赋值
df.loc[:,'D'] = np.array([5] * len(df))
#where 操作赋值.
df2 = df.copy()
df2[df2 > 0] = -df2

#丢失的数据
#pandas主要使用np.nan替换丢失的数据. 默认情况下它并不包含在计算中

#重建索引允许更改/添加/删除指定轴索引,并返回数据副本.
df1 = df.reindex(index=dates[0:4],columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
#删除任何有丢失数据的行.
df1.dropna(how='any')
#填充丢失数据
df1.fillna(value=5)
#获取值是否nan的布尔标记
pd.isnull(df1)



data={'year':[2000,2001,2002,2003],
'income':[3000,3500,4500,6000]}
data1=pd.DataFrame(data,columns=['year','income','outcome'],
index=['a','b','c','d'])
data2=data1.reindex(['a','b','c','d','e'])
print(data2)
data2=data1.reindex(['a','b','c','d','e'],method='ffill')
print(data2)
print(data1.drop(['a']))
print(data1[data1['year']>2001])
print(data1.loc[['a','b'],['year','income']])
print(data1.ix[data1.year>2000,:2])
print(data1.iloc[:,:2][data1.year>2000])#等价于上一句

#dataframe运算
data={'year':[2000,2001,2002,2003],
'income':[3000,3500,4500,6000]}

data1=pd.DataFrame(data,columns=['year','income','outcome'],
index=['a','b','c','d'])
data1['outcome']=range(1,5)
print(data1)

data2=pd.DataFrame(data,columns=['year','income','outcome'],
index=['a','b','c','d'])
data2=data2.reindex(['a','b','c','d','e'])
print(data2)

print(data1.add(data2,fill_value=0))


#对数据进行多维度的索引
#
data = pd.Series(np.random.randn(10), index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'], 
[1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
print(data)
print(data.index)
print(data['c'])
print(data[:,2])
print(data.unstack())
print(data.unstack().stack())


#
data = pd.DataFrame(np.random.randn(10,6), index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'], 
[1, 2, 3, 1, 2, 3, 1, 2, 2, 3]],columns=[['a','a','b','b','c','c'],[1,2,1,2,1,2]])
print(data)
print(data.index)
print(data['c'])
print(data.loc['c'])
print(data.loc['c',['b','c']]) #多种的索引，很难直接取到内层的索引，应使用下面的groupby


data.index.names=['lv1','lv2']
data.columns.names=['cl1','cl2']
print(data.index)
#data1=data.groupby(by='lv2',level='lv2') #或下一句
data1=data.groupby(level='lv2')
print(data1)
for name,group in data1:
    print(name)
    print(group)
    
data1=data.groupby(level='lv1')
print(data1)
for name,group in data1:
    print(name)
    print(group)
    
data1=data.groupby(level='cl1',axis=1)
print(data1)
for name,group in data1:
    print(name)
    print(group)

data1=data.groupby(level='cl2',axis=1)
print(data1)
for name,group in data1:
    print(name)
    print(group)
    
data1=data.groupby('lv1')['b']
for name,group in data1:
    print(name)
    print(group)
    
#data1=data.groupby('lv1')[['b','c']] #这种方式错误可以先取出来然后再group
data1=data[['b','c']].groupby('lv1')
for name,group in data1:
    print(name)
    print(group)
    
'''

#通过一个数组来创建Index对象。在创建的同时我们还可以通过name指定索引的名称
index = pd.Index(['C','D','E','F','G','A','B'], name='note')
#Index对象可以互相之间做集合操作
a = pd.Index([1,2,3,4,5])
b = pd.Index([3,4,5,6,7])
print("a|b = {}\n".format(a|b))
print("a&b = {}\n".format(a&b))
print("a.difference(b) = {}\n".format(a.difference(b)))


#MultiIndex，或者称之为Hierarchical Index是指数据的行或者列通过多层次的标签来进行索引
multiIndex = pd.MultiIndex.from_arrays([
['Geagle', 'Geagle', 'Geagle', 'Geagle',
'Epple', 'Epple', 'Epple', 'Epple', 'Macrosoft',
'Macrosoft', 'Macrosoft', 'Macrosoft', ],
['S1', 'S2', 'S3', 'S4', 'S1', 'S2', 'S3', 'S4', 'S1', 'S2', 'S3', 'S4']],
names=('Company', 'Turnover'))
df = pd.DataFrame(data=np.random.randint(0, 1000, 36).reshape(-1, 12),
index=[2016, 2017, 2018],
columns=multiIndex)
print("df = \n{}\n".format(df))

#print(df.loc[2017, (('Geagle', 'Epple', 'Macrosoft') ,'S1')])#ERROR
#print(df.loc[2018, (['Geagle','Epple'])])#ERROR
#df.loc[2017, (['Geagle', 'Epple', 'Macrosoft'] ,'S1')]#ERROR
#print(df.loc[2017:2018, [('Epple',('S1','S2')),('Geagle','S1')]])#ERROR
#多种的索引只能用tuple把单个的合在一起，比如('Epple','S1')
#print(df.loc[2017, 'Geagle':'Epple'])
print(df.loc[2017, [('Epple','S1'),('Epple','S2'),('Geagle','S1'),('Geagle','S2')]])
print(df.loc[2017, [('Geagle','S1')]])
print(df.loc[2017:2018, [('Geagle','S2')]])
print(df.loc[2017:2018, [('Epple','S1'),('Epple','S2'),('Geagle','S1'),('Geagle','S2')]])



'''
#统计
df.mean()
df.mean(1)
s = pd.Series([1,3,5,np.nan,6,8],index=dates).shift(2)
df.sub(s,axis='index')
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())


data=pd.DataFrame(np.arange(10).reshape((2,5)),index=['c','a'],
columns=['one','four','two','three','five'])
print(data.describe())
print(data.sum())
print(data.sum(axis=1))


#相关系数与协方差
data=pd.DataFrame(np.random.random(20).reshape((4,5)),index=['c','a','b','c'],
columns=['one','four','two','three','five'])
print(data)
print(data.one.corr(data.three))#one和three的相关系数
print(data.one.cov(data.three))#one和three的协方差为
print(data.corrwith(data.one))#one和所有列的相关系数


#直方图
s = pd.Series(np.random.randint(0,7,size=10))
s.value_counts()


#字符串方法
#序列可以使用一些字符串处理方法很轻易操作数据组中的每个元素,比如以下代码片断。 
#注意字符匹配方法默认情况下通常使用正则表达式（并且大多数时候都如此）. 
#更多信息请参阅字符串向量方法.
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()

s1 = pd.Series([' 1', '2 ', ' 3 ', '4', '5']);
print("s1.str.rstrip():\n{}\n".format(s1.str.lstrip()))
print("s1.str.strip():\n{}\n".format(s1.str.strip()))
print("s1.str.isdigit():\n{}\n".format(s1.str.isdigit()))

s2 = pd.Series(['Stairway to Heaven', 'Eruption', 'Freebird',
'Comfortably Numb', 'All Along the Watchtower'])
print("s2.str.lower():\n{}\n".format(s2.str.lower()))
print("s2.str.upper():\n{}\n".format(s2.str.upper()))
print("s2.str.len():\n{}\n".format(s2.str.len()))
'''


#数据整合
df1 = pd.DataFrame({'Note': ['C', 'D'],
'Weekday': ['Mon', 'Tue']},
index=[1, 2])

df2 = pd.DataFrame({'Note': ['E', 'F'],
'Weekday': ['Wed', 'Thu']},
index=[1, 4])

df3 = pd.DataFrame({'Note': ['G', 'A', 'B'],
'Weekday': ['Fri', 'Sat', 'Sun']},
index=[1, 6, 7])

df_concat = pd.concat([df1, df2, df3], keys=['df1', 'df2', 'df3'])
print("df_concat=\n{}\n".format(df_concat))
print(df_concat.loc[:,['Note','Weekday']])
print(df_concat.loc['df1',['Note','Weekday']])
print(df_concat.loc[['df1','df2'],['Note','Weekday']])
print(df_concat.loc[(['df1','df2'],1),['Note','Weekday']])


df = pd.DataFrame(data=np.random.randint(0, 1000, 48).reshape(-1, 12),
index=[['pa','pa','pb','pb'],[2015,2016, 2017, 2018]],columns=[
['Geagle', 'Geagle', 'Geagle', 'Geagle',
'Epple', 'Epple', 'Epple', 'Epple', 'Macrosoft',
'Macrosoft', 'Macrosoft', 'Macrosoft', ],
['S1', 'S2', 'S3', 'S4', 'S1', 'S2', 'S3', 'S4', 'S1', 'S2', 'S3', 'S4']])
print(df)
print(df[['Epple','Macrosoft']].loc['pa'])
print(df[[('Epple','S2'),('Macrosoft','S1')]].loc['pa'])
#行列不能同时取得情况下，可以先取一边，然后再取一边

#merge
df1 = pd.DataFrame({'key': ['K1', 'K2', 'K3', 'K4'],
'A': ['A1', 'A2', 'A3', 'A8'],
'B': ['B1', 'B2', 'B3', 'B8']})

df2 = pd.DataFrame({'key': ['K3', 'K4', 'K5', 'K6'],
'A': ['A3', 'A4', 'A5', 'A6'],
'B': ['B3', 'B4', 'B5', 'B6']})

print("df1=n{}n".format(df1))
print("df2=n{}n".format(df2))

merge_df = pd.merge(df1, df2)
merge_inner = pd.merge(df1, df2, how='inner', on=['key'])
merge_left = pd.merge(df1, df2, how='left')
merge_left_on_key = pd.merge(df1, df2, how='left', on=['key'])
merge_right_on_key = pd.merge(df1, df2, how='right', on=['key'])
merge_outer = pd.merge(df1, df2, how='outer', on=['key'])

print("merge_df=\n{}\n".format(merge_df))
print("merge_inner=\n{}\n".format(merge_inner))
print("merge_left=\n{}\n".format(merge_left))
print("merge_left_on_key=\n{}\n".format(merge_left_on_key))
print("merge_right_on_key=\n{}\n".format(merge_right_on_key))
print("merge_outer=\n{}\n".format(merge_outer))

#join
df3 = pd.DataFrame({'key': ['K1', 'K2', 'K3', 'K4'],
'A': ['A1', 'A2', 'A3', 'A8'],
'B': ['B1', 'B2', 'B3', 'B8']},
index=[0, 1, 2, 3])

df4 = pd.DataFrame({'key': ['K3', 'K4', 'K5', 'K6'],
'C': ['A3', 'A4', 'A5', 'A6'],
'D': ['B3', 'B4', 'B5', 'B6']},
index=[1, 2, 3, 4])

print("df3=\n{}\n".format(df3))
print("df4=\n{}\n".format(df4))

join_df = df3.join(df4, lsuffix='_self', rsuffix='_other')
join_left = df3.join(df4, how='left', lsuffix='_self', rsuffix='_other')
join_right = df1.join(df4, how='outer', lsuffix='_self', rsuffix='_other')

print("join_df=\n{}\n".format(join_df))
print("join_left=\n{}\n".format(join_left))
print("join_right=\n{}\n".format(join_right))


#数据集合和分组操作

#groupby将数据分组，分组后得到pandas.core.groupby.DataFrameGroupBy类型的数据。
#agg用来进行合计操作，agg是aggregate的别名。
#apply用来将函数func分组化并将结果组合在一起。

df = pd.DataFrame({
'Name': ['A','A','A','B','B','B','C','C','C'],
'Data': np.random.randint(0, 100, 9)})
print('df=\n{}\n'.format(df))

groupby = df.groupby('Name')

print("Print GroupBy:")
for name, group in groupby:
    print("Name: {}\nGroup:\n{}\n".format(name, group))

print(groupby.sum())
print(groupby.agg(['sum']))
print(groupby.agg([('Total', 'sum'), ('Min', 'min')]))

def sort(df):
    return df.sort_values(by='Data', ascending=False)

print("Sort Group: \n{}\n".format(groupby.apply(sort)))

'''
#合并
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)


#连接
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})


#添加
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = df.iloc[3]
df.append(s, ignore_index=True)


#分组
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
'C' : np.random.randn(8),
'D' : np.random.randn(8)})
df.groupby('A').sum()
df.groupby(['A','B']).sum()


#重塑
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two','one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]

#堆叠
#被“堆叠”数据桢或序列(有多个索引作为索引), 其堆叠的反向操作是未堆栈, 上面的数据默认反堆叠到上一级别:
stacked = df2.stack()
stacked.unstack()


#数据透视表

df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
'B' : ['A', 'B', 'C'] * 4,
'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
'D' : np.random.randn(12),
'E' : np.random.randn(12)})
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


#时间序列

rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min', how='sum')

rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)

ts_utc = ts.tz_localize('UTC')

ts_utc.tz_convert('US/Eastern')

rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ps = ts.to_period()
ps.to_timestamp()

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()
'''

#时间相关
#某个具体的时间点（Timestamp），例如：今天下午一点整
#某个时间范围（Period），例如：整个这个月
#某个时间间隔（Interval），例如：每周二上午七点整

#打印了今天的日期，并通过timedelta进行了日期的减法运算
now = dt.datetime.now();
print("Now is {}".format(now))

yesterday = now - dt.timedelta(1);
print("Yesterday is {}\n".format(yesterday.strftime('%Y-%m-%d')))

#借助pandas提供的接口，我们可以很方便的获得以某个时间间隔的时间序列
this_year = pd.date_range(dt.datetime(2018, 1, 1),
dt.datetime(2018, 12, 31), freq='5D')
print("Selected days in 2018: \n{}\n".format(this_year))

df = pd.DataFrame(np.random.randint(0, 100, this_year.size), index=this_year)
print("Jan: \n{}\n".format(df['2018-01']))




'''
#分类
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
df["grade"].cat.categories = ["very good", "good", "very bad"]

df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])

df["grade"]
df.sort_values(by="grade")
df.groupby("grade").size()

#绘图
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')
'''

plt.figure()
data=pd.DataFrame([1,2,3,4,3,2,1])
data.hist(bins=10)

plt.figure()
plt.plot(np.random.randn(50).cumsum(), 'k--')

plt.figure()
plt.hist(np.random.randn(100), bins=20, color='k',alpha=0.3)

plt.figure()
plt.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.hist(np.random.randn(100), bins=20, color='k',alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))


#获取数据输入/输出
'''
df.to_csv('foo.csv')
pd.read_csv('foo.csv')

df.to_hdf('foo.h5','df')
pd.read_hdf('foo.h5','df')

df.to_excel('foo.xlsx', sheet_name='Sheet1')
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

#read_csv
#read_table
#read_fwf
#read_clipboard
#read_excel
#read_hdf
#read_html
#read_json
#read_msgpack
#read_pickle
#read_sas
#read_sql
#read_stata
#read_feather
'''







