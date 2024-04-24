#!/usr/bin/env python
# coding: utf-8

# In[13]:


# empty series
import pandas as pd
ser = pd.Series()
print(ser)


# In[3]:


conda install pandas


# In[14]:


# single element in series 
ser = pd.Series(5)
print(ser)


# In[22]:


# give the datatype (scalar series means one element)
ser = pd.Series(5 , dtype = float)
print(ser)


# In[28]:


# multiple elements in series
ser = pd.Series([90,28,63])
print(ser)


# In[2]:


# using random function generate elements in series
print(pd.Series(np.random.randint(11,21,5)))


# In[21]:


# by using lst
lst = [1,2,3]
a = pd.Series(lst)
print(a)


# In[34]:


# list convert to dataframe
l1 = ['abc','def','ghi']
l2 = [67,98,78]
df = pd.DataFrame([l1,l2]).T
df.columns = ['names','marks']
df.index = ['stud_1','stud_2','stud_3']

df


# In[37]:


# nested list
list = [['abc','def','ghi'],[67,98,78]]
df = pd.DataFrame(list)
df


# In[40]:


# marks not defined so it takes default value float
ser = pd.Series(marks,index =['s1','s2','s3','s4','s5'])

print(ser)


# In[41]:


# random numbers generates default index value takes
print(pd.Series(np.random.randint(11,21,5)))


# In[3]:


# marks taken as random
import pandas as pd
import numpy as np
marks = np.random.randint(1,100,5)
ser = pd.Series(marks,index =['s1','s2','s3','s4','s5'])
print(ser)


# In[27]:


# marks should be fix
marks = np.repeat(5,5)
ser = pd.Series(marks,index =['s1','s2','s3','s4','s5'])
print(ser)


# In[6]:


# marks given
marks =[59,60,78,78,90]
ser = pd.Series(marks,index =['s1','s2','s3','s4','s5'])
print(ser)


# In[8]:


# create DataFrame
import pandas as pd
df = pd.DataFrame()
print(df)


# In[9]:


# 1-d arr convert to data frame(data frame is a 2 dimensional)
import pandas as pd
arr =[11,22,33,44]
df = pd.DataFrame(arr)
df


# In[11]:


# list convert to data frame(data frame is a 2 dimensional)
import pandas as pd
list =[67,34,89,12]
df = pd.DataFrame(list)
df


# In[1]:


# list of strings convert to DataFrame 
import pandas as pd
list =['manu','hari','teja','paduu']
df = pd.DataFrame(list)

df.columns = ['names']
df.index = [1,2,3,4]

df

# adding column
df['marks'] = pd.Series([67,89,85,78])
df

df['stud id'] = pd.Series([101,102,103,104])
df

# delete the col name(del)
del df ['names']
df
# delete the col name(pop)
df.pop('stud id')
df


# In[17]:


# 2-d arr convert to data frame(data frame is a 2 dimensional)
import pandas as pd
import numpy as np
arr =np.array([[11,22,33,44],[55,78,99,12]])
df = pd.DataFrame(arr)
df.index=['stud1','stud2']
df.columns = ['tel','eng','sci','math']
df


# In[3]:


# dictionary convert to series
import pandas as pd
dic = {'a':10,'b':20,'c':30}
ser = pd.Series(dic)
ser
# convert dataframe
df = pd.DataFrame(ser)
df


# In[20]:


# 2 method(covert dic to dataframe)
dic = {'a':10,'b':20,'c':30}
df = pd.DataFrame([dic]).T
df


# In[19]:


# convert dic to dataframe (first we convert dic into series after it convert dataframe)(1 method)(key value is row name)
dic = {'s1':1,'s2':2,'s3':3,'s4':4,'s5':5}
ser1 = pd.Series(dic)
# covert dataframe
df = pd.DataFrame(ser1)
df.columns = ['names']
# df.index = [101,102,103,104,105]
df



# In[1]:


# dictionary 
data = {'name':['a','b','c'] ,'age': [10,11,12]}
print(data)
# convert dataframe
import pandas as pd
df  = pd.DataFrame(data)
df


# In[7]:


help(pd.Series)


# In[10]:


import pandas as pd
lst = ['a','d','c']
df = pd.DataFrame(lst,index=['s1','s2','s3'],columns =['names'])
print(df)


# In[39]:


import pandas as pd
import numpy as np

arr=np.array([1,2,3,4,])
#arr to df
df = pd.DataFrame(arr)
# arr to ser
arrser = pd.Series(arr)
print(arrser)
df


# In[46]:


pip install names


# In[49]:


# create dataframe with student data
studid = np.arange(1,11)
print('STUDENT ID:\n',studid)
import names
name = []
for i in range(10):
    name.append(names.get_full_name(gender = 'female'))
print('STUDENT NAMES:\n',name)
marks = np.random.randint(360,580,10)
print('marks:\n',marks)
perc = ((marks/600)*100)
# perc = np.round(((marks/600)*100),2)
print('percentage:\n',perc)
import pandas as pd
df = pd.DataFrame([studid,name,marks,perc]).T
df.columns = ['stud id','stud name','marks','percentage']
df.index = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
df


# In[172]:


#df.loc['en1']
# df.iloc[0]


# In[173]:


# attributes perform df
df.T


# In[2]:


df.axes


# In[165]:


df.dtype


# In[145]:


df.size


# In[146]:


df.shape


# In[147]:


df.values


# In[4]:


df_3.head()


# In[151]:


df.tail(2)


# In[153]:


# BASIC STATISTICS
df.count()


# In[154]:


df.count(1)


# In[155]:


df.count(0)


# In[9]:


df_3['sal'].sum()
# MEAN
 #df.sal.mean() # one way
    
#df['sal'].mean() # another way
import numpy as np
#np.mean(df_3.sal)    # another way


# In[165]:


df.sal.prod() # product


# In[166]:


# cummulative product
df.sal.cumprod() # it multiply sequence order like 1x2,2x3,6x3,18x4


# In[167]:


df.sal.abs() # absolute devation


# In[168]:


df.sal.std() # standard devation


# In[169]:


df.sal.var() # variance


# In[116]:


# INDEXIBG & SELECTINGdf_3
# single column indexing 
df
# df_3.sal[2:] # (2 column start until the columns end)
# df_3.sal[0:]
# df_3.sal[:2]   # (0 column start until the number u given -1 columns)

# single column selecting

# a = df_3.sal[1:5]
# a
# df_3.sal[2:4]

# 2 columns indexing ad selecting
#df_3[['sal','empname']][0:2]

# covert to df
df = df[['fname','lname']][0:3]
df


# In[8]:


#iloc and loc
#df_3
#iloc(give the index number you get all columns and entire row )
# df_3.iloc[2]
# df_3.iloc[0:2,1:3] # 0:2 represents rows , 1: represents columns(specific col)
# df_3.iloc[0:2,[2,1]]
# create sub df
#df_3.iloc[0:2,1:3] # index number column number

#loc(give the index name)
df_3.loc['emp1']
# create sub df
# df_3.loc[['emp1','emp4'],['empname','sal']] # index name column name


# In[2]:


# file uploading to python
# csv file import
import pandas as pd
df_4 = pd.read_csv(r'C:\Users\ACER\PycharmProjects\pythonProject11\venv\Lib\site-packages\matplotlib\mpl-data\sample_data\percent_bachelors_degrees_women_usa.csv')
print(df_4)


# In[3]:


df_4.head()
df_4.tail()
df_4.count()
df_4.info()
df_4.sum()


# In[17]:


import pandas as pd
df_4 = pd.read_table(r'C:\Users\ACER\anaconda3\Library\RELEASE.txt')
print(df_4)


# In[178]:


# create dataframe by using dic
import pandas as pd
import numpy as np
dic = {'id':[1,2,3,4,5],'fname':['tej','manu','hari','paduu','asw'],'lname':['sab','racha','chil','kota','tham'],
        'dob':[19/1/2001,24/5/2000,5/9/2000,4/6/2001,8/6/2001],'ethnicity':['white','black','brown','white','black'],
        'gender':['f','f','f','f','f'],'acdameic entry':[2004,2005,2003,2006,2004],'gpa':[7.8,9.7,5.6,7.8,5.6]}
df  = pd.DataFrame(dic)
df.index = ['en1','en2','en3','en4','en5']
df


# In[66]:


# create dataframe by using list
l1 = [1,2,3,4]
l2 = ['t','m','h','p']
l3 = ['sab','rac','chil','kot']
l4 = ['jan_19','may_24','april_7','sep_3']
df = pd.DataFrame([l1,l2,l3,l4])
df.columns = ['id','fname','lname','dob']
df.index = ['en1','en2','en3','en4']
df


# In[69]:


# create dataframe by using array
import pandas as pd
import numpy as np
arr = np.array([[1,2,3,4],['abc','def','ghi','jkl'],['jan','feb','mar','apr'],['m','f','m','f'],['bvrm','vjy','vzg','dtmi']])
df = pd.DataFrame(arr)
df


# In[ ]:


arr = np.ndarray(size = (2,3) )
print("enter the elements")


# In[70]:


import pandas as pd
import numpy as np
arr = np.zeros((8,8), dtype = int)
arr[1::2,::2]=255
arr[::2,1::2]=255
arr



# In[73]:


# chess board pattern
arr = np.zeros((8,8), dtype = int)
arr[1::2,::2]=255
arr[::2,1::2]=255
arr

# given size to print chessboard pattern

a = int(input("enter the size:"))
size = arr[:a,:a]
print('sub matrix of arr:\n',size)




# In[83]:


import numpy as np
arr = np.zeros((8,8), dtype = int)
arr[1::2,::2]=255
arr[::2,1::2]=255
arr

a = int(input("enter the size:"))
con = np.where(a%2!=0)
print("enter the even number only")
size = arr[:a,:a]
print('sub matrix of arr:\n',size)





# In[98]:


# MISSING DATA
# cretae database using random
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randint(35,100,30).reshape(6,5),index = ['s1','s3','s5','s7','s9','s10'] , 
                  columns = ['eng','math','sci','soc','hin'])
df


# In[99]:


import pandas as pd


# In[10]:


# reindexing(add the rows in existing rows)

df_1 = df.reindex(['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10'])
df_1


# In[14]:


# isna( the value is flase,nan is true) notnull(the values is true,nan is flase)
df_1.isna()
#df_1.notnull()


# In[15]:


# sum of individual column
df_1['eng'].sum()
df['math'].sum()


# In[49]:


# all nan values replace with the number you can give
df_1.fillna(28)


# In[51]:


# pad (the nan values replace with previous row data)
df_1.fillna(method = 'pad')


# In[52]:


#  bfill(the nan values replace with backward row data)
df_1.fillna(method = 'bfill')


# In[54]:


# applying mean to all columns(with out nan rows)
df.mean()
df_1.mean()


# In[55]:


# appliying mean individually column
np.mean(df_1['eng'])


# In[56]:


# fill missing data by individually column mean
# applying mean on particular column that mean replace with all nan values
print(df_1['eng'].fillna(np.mean(df.eng)))


# print(df_1)
# 

# In[8]:


# fill missing data by all column mean
# applying mean on all column that mean replace with all nan values
df_1.fillna(np.mean(df_1))


            


# In[10]:


# drop the all nan values
df_1.dropna()


# In[16]:


# replace

df_1.replace({56:39}) # replace 39 in all where 56 is located in df
# replace particular column value where 77 is replaced with 80  
df_1['eng'].replace({77:80})


# In[3]:


# GROUPBY
# emp  dataframe is created
import pandas as pd
emp = {'empid':[101,102,103,104],'empname':['damon','stefan','elena','caroline'],'year':[2016,2018,2017,2019],
          'sal':[25000,30000,28000,20000]}
df_3 = pd.DataFrame(emp)
df_3.index = ['emp1','emp2','emp3','emp4']
df_3


# In[21]:


df_3.groupby(['empname']).groups


# In[22]:


df_3.groupby(['empid']).groups


# In[23]:


df_3.groupby(['year']).groups


# In[24]:


df_3.groupby(['sal']).groups


# In[62]:


# create sub dataframe in main dataframe 
gs = df_3.groupby(['empid'])
gs.groups
for key,values in gs:
    print(key)
    print(values)
    print()
    
# sal df

gs = df_3.groupby(['sal'])
gs.groups
for key,values in gs:
    print(key)
    print(values)


# In[63]:


# how to get some data in df
gs = df_3.groupby(['year'])
year = gs.get_group(2016)  # In Pranthesis whatever you want coln name is given
year


# In[66]:


gs = df_3.groupby(['empid'])
gs.groups
for key,values in gs:
    print(key)
    print(values)
empid = gs.get_group(104)
empid


# In[170]:


gs = df_3.groupby(['sal'])
gs.groups
sal = gs.get_group(20000)
sal


# In[171]:


gs = df_3.groupby(['empname'])
gs.groups
empname = gs.get_group('stefan')
empname


# In[176]:


# how to apply aggregate functions
print(df)
df.groupby('sal').agg([np.sum])


# In[173]:


# apply all agg functions
df_3.groupby('sal').agg([np.mean,np.sum,np.std])


# In[174]:


# we can apply describe on df
empname.describe()


# In[32]:


print(df_3)
year.describe()


# In[134]:


# TRANSFORMATION
gs = df_3.groupby('year')
for key,value in gs:
    print(key)
    print(values)
scale = lambda x : x-x.mean()
gs.transform(scale)


# In[137]:


# FLITRATION
gs.filter(lambda x : len(x)<3)


# In[111]:


import numpy as np
import pandas as pd
ser = pd.Series(np.random.randint(1,100,30))
print(ser)


# In[112]:


a = np.min(ser)
a


# In[103]:


b = np.max(ser)
b


# In[106]:


c = np.percentile(ser,25)
c


# In[116]:


#d = np.percentile(ser, 50)
d = np.median(ser)
d


# In[115]:


e = np.percentile(ser, 75)
e


# In[113]:


print('the series of min , max , 50th , 75th , 25th percentile is:\n',a,b,c,d,e)


# In[125]:


# frequency count of unique items(variable name.value_counts())
import pandas as pd
import numpy as  np
ser_1 = pd.Series(np.random.randint(1,100,30))
 #ser_1 = pd.Series([1,2,3,1,2,4,5,7,9,5,4,1,7,3])
 #print(ser_1)
a = ser_1.value_counts()
print('the frequency count of unique items:\n',a)


# In[187]:


import pandas as pd
import numpy as np
ser = pd.Series([1,2,3,4,6,8,9,10,15,12,18,15,21,17,24,19,27,20,30])
a = np.where(ser%3 == 0)
print('the positions of 3 multiple is:\n',a)


# In[189]:


import pandas as pd
import numpy as np
serA = pd.Series([10,9,6,5,3,1,12,8,13])
serB = pd.Series([1,3,10,13])
print('serA is:\n',serA)
print('serB is:\n',serB)
for i in serB:
    a = [pd.Index(serA).get_loc(i)]
    print('the common elements position of series b in series a:\n',a)


# In[200]:


import pandas as pd
truth = pd.Series([10,11,33,98,22,16])
predicted = pd.Series([10,33,45,67,12,11])

error = (truth-predicted)

squared = (error)^2
print('squared value is:',squared)
print()

print('the error is:',error)
print()

mse = mean_squared_error(truth.predicted)
print('mean squared error is:',mse)


# In[40]:


# diagonal matrix
y = np.diag([[2,4,5],[6,3,2],[1,4,6]])
 #z = np.diag(y)
print(y)


# In[26]:


b = np.random.randint(1,100, size =(2,3))
b


# In[6]:


# 
import numpy as np
a = np.ndarray(5, dtype = int)
for i in range(5):
    a[i] = int(input())
print(a)


# In[27]:


import numpy as np
a = np.array([10,12,14,15])
print(a)
print(a.ndim)


# In[53]:


# merge
import pandas as pd

#left = pd.DataFrame({'col_1' : np.arange(40,50), 'col' : ['a','b','c','d','e','f','g','h','i','j']})

left = pd.DataFrame({'col_1': np.random.randint(20,30,10), 'col' : [1,2,3,4,5,6,7,8,9,10]})
print('the first df:\n',left)

#right = pd.DataFrame({'col_2' : np.arange(50,60), 'col' : ['a','b','c','d','e','f','g','h','i','j']})

right =pd.DataFrame({'col_2': np.random.randint(40,60,10), 'col' : [1,2,3,4,5,6,7,8,9,10]})
print('the second df:\n',right)

pd.merge(left,right)


# In[112]:


# concat
print('concating two or more dataframes with  index:\n')
pd.concat([left,right])


# In[106]:


print(pd.concat([left,right],axis=1))


# In[104]:


print(pd.concat([left,right],axis=0))


# In[105]:


print(pd.concat([left, right], axis = 0,keys = ['gruop A','group B']))


# In[118]:


# it gives continuous index 
# print(pd.concat([left, right], axis = 0, ignore_index = True)) #1
print(pd.concat([left, right], axis = 0, ignore_index = 1))


# In[119]:


#print(pd.concat([left, right], axis = 0,ignore_index = False)) 
print(pd.concat([left, right], axis = 0, ignore_index = 0))


# In[122]:


print(left.append(right))


# In[17]:


# merge,concat and join operations

# create left_df
import numpy as np
import pandas as pd
stud_id = np.arange(1001,1011)
print(stud_id)
import names
name = []
for i in range(10):
    name.append(names.get_full_name(gender = 'male'))
print(name)
marks = np.random.randint(500,590,10)
print(marks)
courses = np.take(list('ABCD'),np.random.randint(3,size =10))
print(courses)
left = {'stud_id':pd.Series(stud_id),
       'name':pd.Series(name),
       'marks':pd.Series(marks),
       'courses':pd.Series(courses)}
left_df = pd.DataFrame(left)
left_df


# In[15]:


# create right df
import numpy as np
import pandas as pd
stud_id = np.arange(1005,1015)
print(stud_id)
import names
name = []
for i in range(10):
    name.append(names.get_full_name(gender = 'female'))
print(name)
marks = np.random.randint(500,590,10)
print(marks)
courses = np.take(list('ABCD'),np.random.randint(3,size =10))
print(courses) 
right = {'stud_id':pd.Series(stud_id),
       'name':pd.Series(name),
       'marks':pd.Series(marks),
       'courses':pd.Series(courses)}
right_df = pd.DataFrame(right)
right_df


# In[21]:


# merge left_df and right_df
pd.merge(left_df,right_df)


# In[22]:


pd.merge(left_df,right_df,on ='stud_id')


# In[23]:


# based on 2 common column

pd.merge(left_df,right_df,on = ['stud_id','courses'])


# In[134]:


# joins
#left join(only left values display right values should be nan)
pd.merge(left_df,right_df, on ='student_id', how = 'left')


# In[203]:


# right join(left values should be nan right values display)
pd.merge(left_df,right_df,on = 'stud_id', how ='right')


# In[24]:


# inner nothing but the merge on 1 commom column
pd.merge(left_df,right_df,on = 'stud_id', how ='inner')


# In[157]:


pd.merge(left_df,right_df, on=['student_id'],how='outer')


# In[8]:


# pivot table
import pandas as pd
pt = pd.DataFrame({'empid' : np.random.randint(100,110,10),'working_days' : np.random.randint(1,31,10),
                   'desg':['account','professor','manager','asstmng','asstprof','junract','sales','doctor','police','lawyer'],
                  'sal':[20000,35000,25000,15000,22000,60000,34000,32000,20000,25000]})
pt


# In[131]:


pt.pivot(index = 'desg',columns = 'working_days',values = 'sal')


# In[132]:


pt.pivot(index = 'working_days',columns = 'desg',values = 'empid')


# In[133]:


table = pd.pivot(pt,index = 'sal',columns = 'desg' , values = 'working_days')
table


# In[10]:


np.sum(pt['sal'])


# In[12]:


np.mean(pt['sal'])


# In[13]:


np.median(pt['working_days'])


# In[23]:


np.var(pt['sal'])


# In[24]:


np.std(pt['working_days'])


# In[3]:


# cross table
import pandas as pd
ct = pd.read_csv(r'C:\Users\ACER\OneDrive\Pictures\Screenshots\Documents\usa women.csv')
ct


# In[115]:


import pandas as pd
from seaborn import load_dataset
cross = load_dataset('ct')
cross


# In[7]:


ct.head()


# In[16]:


df = ct.head(12)
df


# In[41]:


pd.crosstab(df['Year'],df['Computer Science'])
# pd.crosstab(df['Year'],df['Computer Science'],margins = True)
# pd.crosstab(df['Year'],df['Computer Science'],normalize = 'index',margins = True) # normalize reduce the values 0 to 1


# In[43]:


pd.crosstab(df['Year'],df['Computer Science'],df['Biology'],aggfunc = 'mean')


# In[51]:


df.groupby(['Year','Computer Science']).size()


# In[54]:


# unstack
df_stacked = df.unstack()
df_stacked


# In[55]:


# stack
df_stacked_1 = df.stack()
df_stacked_1


# In[65]:


# melt
# df_melt = df.melt()
# df_melt
df_melt.head(10)


# In[58]:


df.info()


# In[71]:


# categorical data
# 1st method
import pandas as pd
l1 = pd.Series(['a','b','c','d'], dtype = 'category') # four objects
l1


# In[72]:


import pandas as pd
l1 = pd.Series(['a','b','c','a'], dtype = 'category') # three objects a is repeates it count as 1 
l1


# In[73]:


# 2nd method
cat = pd.Categorical(['a','v','s','r','t']) # 5 objects
cat


# In[74]:


cat = pd.Categorical(['a','v','s','r','t'],['a','v','r','t']) # first[] is alphabets defined second[] is object defined so it 4
cat


# In[111]:


cat = pd.Categorical(['a','b','c','d','e'],['a','b','d','c'],ordered =True)
cat


# In[112]:


day = pd.Categorical(['mon','sun','tues','wed'])
day


# In[116]:


day.Categories


# In[ ]:





# In[102]:


# comparison between categorical data
c1 = pd.Categorical([1,2,3],categories=[1,2,3])
c2 = pd.Categorical([1,2,3],categories=[1,2,3])
print(c1==c2)


# In[136]:


# working with time series
import pandas as pd
from datetime import datetime
# date_range = pd.date_range(start = '1/1/2022', end = '1/3/2022')
date_range = pd.date_range(start = '1/1/2022', end = '1/3/2022', freq='h') # H-hours refer
date_range


# In[137]:


date_range = pd.date_range(start = '1/1/2022', end = '1/3/2022', freq='t') # T-seconds refer 
date_range


# In[135]:


date_range = pd.date_range(start = '1/1/2022', end = '3/3/2022',freq='d') # d-days refer 
date_range


# In[146]:


date_range = pd.date_range(start = '1/1/2022', end = '6/3/2022',freq='w') # w-weeks refer 
date_range


# In[148]:


date_range = pd.date_range(start = '1/1/2022', end = '12/3/2022',freq='m') # m-months refer 
date_range


# In[150]:


date_range = pd.date_range(start = '1/1/2022', end = '5/3/2022',freq='b') # b-business days refer 
date_range


# In[157]:


date_range = pd.date_range('1/1/2022', periods = 10) # periods-how many dyas you want refer 
date_range


# In[4]:


# writing with text data
import pandas as pd
import names
name = []
for i in range(10):
    name.append(names.get_full_name(gender = 'female'))
print(name)
ser = pd.Series(name)
ser


# In[161]:


ser.str.lower()


# In[162]:


ser.str.upper()


# In[164]:


ser.str.isnumeric()


# In[167]:


ser.str.islower()


# In[168]:


ser.str.isupper()


# In[169]:


ser.str.len()


# In[171]:


ser.str.count('a') # how many times a repeated give count


# In[5]:


ser.str.split(' ')


# In[7]:


ser.str.cat(sep = '#') # seperater symbol in white place-refer


# In[18]:


ser.str.get_dummies()


# In[15]:


ser.str.contains('a') # letter contains or not -refer


# In[16]:


ser.str.replace('e','@')


# In[17]:


ser.str.repeat(2) # repaet how many no of times


# In[21]:


ser.str.startswith('S')


# In[22]:


ser.str.endswith('l') 


# In[23]:


ser.str.find('a') # find a in names how many times repeated gives number


# In[24]:


ser.str.findall('e') # find e in all names and how many times present


# In[25]:


ser.str.swapcase() # swap first letter capital be small rest of the letters small be big


# In[33]:


# writing files
import numpy as np
import indian_names
quant = np.random.randint(123,345,10)
verb = np.random.randint(230,599,10)
score = quant+verb
print('gre score is:\n',score)


# In[29]:


pip install indian-names


# In[35]:


pip install os


# In[2]:


import pandas as pd
a = pd.read_csv(r'C:\Users\ACER\Downloads\survey.csv')
a


# In[29]:


# current working directory
import os

cwd = os.getcwd()
print('Current working directory:', cwd)



# In[28]:


import pandas as pd

data = {'name': ['John', 'Jane', 'Sam'], 'age': [25, 30, 21]}
df = pd.DataFrame(data)

df.to_csv('people.csv')
df


# In[59]:


import pandas as pd
graph = pd.read_csv(r'C:\Users\ACER\Downloads\std.csv')
graph


# In[93]:


import pandas as pd
graph = pd.read_csv(r'C:\Users\ACER\Downloads\std.csv')
graph
#graph.columns = [1,2,3,4,5,6,7,8,9,10,11]
#graph


# In[202]:


pie = graph.head()
pie


# In[101]:


# line graph
#graph.plot(y=3)
#graph.plot(y=4,figsize=(10,5))
graph.plot(y=[2,3],figsize=(10,4),title='AGE vs APPROXHEIGHT', ylabel = 'y-axis', xlabel = 'x-axis',color = 'red')


# In[108]:


graph.plot(kind = 'line',y = [4,2,3,5,6,8,9],figsize=(10,5))


# In[114]:


# bar graph
#graph.plot(kind = 'bar', figsize = (20,10), y = 3)
graph.plot(kind = 'bar', figsize = (20,10), y = 3,title = 'APPROXHEIGHT', ylabel = 'y-axis', xlabel = 'x-axis')


# In[198]:


graph.plot(kind = 'bar', figsize = (20,10), y = [3,5,2],title = 'APPROXHEIGHT', ylabel = 'y-axis', xlabel = 'x-axis')


# In[124]:


#histogram

# single column
graph['Age'].plot(kind='hist',bins=20)

# figsize , color, bins
graph['Age'].plot(kind='hist',bins=20,figsize=(10,5),color='pink')

# alpha - reduce color brightness
graph['Age'].plot(kind='hist',bins=20,figsize=(10,5),color='red',alpha=0.5)

# title - lables don't work in histogram
graph['Age'].plot(kind='hist',bins=20,figsize=(10,5),color='pink',alpha =0.5,title = 'AGE')


# In[131]:


graph[['Age','ApproxHeight','ApproxWeight']].plot(kind='hist',bins=20,figsize=(10,5),color='blue',alpha=0.3)


# In[147]:


# box graph

graph['ApproxWeight'].plot(kind = 'box')
#graph['Age'].plot(kind = 'box')



# In[209]:


graph[['ApproxWeight','ApproxHeight']].plot(kind = 'box',vert = True)


# In[165]:


# area
graph.plot(kind = 'area',y=4,color = 'yellow',alpha = 0.5)


# In[221]:


#pie plot

#pie.plot(kind='pie',y='Age')

# autopct refers the values in pie
# pie.plot(kind='pie',y = 'Age',autopct = '%f')

#
pie.plot(kind='pie',y = 'Age',autopct = '%f',legend = 'F')


# In[174]:


# scatter
# it has x,ycolumns 
graph.plot(kind='scatter',x='Age',y='Yr_JoinCampus',title='AGE VS APPROXHEIGHT',figsize=(10,6))


# In[189]:


# hexbin

#
#graph.plot(kind='hexbin',x='Age',y='Yr_JoinCampus')

# grid size
#graph.plot(kind='hexbin',x='Age',y='Yr_JoinCampus',gridsize = 7)

# title 
graph.plot(kind='hexbin',x='Age',y='Yr_JoinCampus',title='AGE VS APPROXHEIGHT',gridsize=(6))


# In[ ]:




