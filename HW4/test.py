import random
import numpy as np
import math

# for _ in range(10):
    # print(random.uniform(0, 1))

# a = numpy.array([[0,0,0],[1,1,1]])
# print(a[0])
# print(a[0,:])
# print(a[0:])



# arr=np.zeros(6)
# # x=np.arange(-2.4, 2.4,4.8/7)
# x=np.linspace(-2.4,2.4,num=7,endpoint=0)
# print(x)
# x=np.array_split(x,7)
# print(x)
# for i in range(1,len(x)):
#     arr[i-1]=x[i][0]
# print(arr)
# a=np.zeros((5,5))
# # a.append(0)
# print(a)

# arr=np.zeros( 4 )
# x=np.arange(0, 10)
# x=np.array_split(x,5)
# for i in range(1,len(x)):
#     arr[i-1]=x[i][0]

# print(arr)

# print(np.digitize( np.array([2,4, 6, 8]),bins=[5]))
# i=np.searchsorted(np.array([2,4,6,8]), 4,side='right')
# print(i)
# print(6 if i==3 else 7)

# t = [[]] * 5
# print(t)
# t[0].append(5)
# t[0].append(2)
# print(t)
# print(len(t))

# i=5
# print(lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25))))

# arr=np.zeros((3,3,3,3))
# a=np.zeros((3,3,3,3))
# # print(a)
# # # print(arr<a)
# l=[2,2,2,2]
# # l.append(1)
# print(type(l))
# print(arr[l])

# mylist = [0, 1, 1]
# x = all(mylist)
# print((arr<=a).all())

# # import os
# # print(os.path.exists("./Tables/cartpole_table.npy"))

# q=np.load("./Tables/cartpole_table.npy")
# print(a.type)
# print(np.array([7.5,6.9]))
# print( np.linalg.norm(np.array([7.5,6.9]),2))

# l=[1,2,6,-1,9]
# print(np.argmin(l))

a=np.array([
    [[1,3,4,5,6],
     [1,3,4,5,7],
     [1,3,4,5,8],],
    [[1,3,4,5,3],
     [1,3,4,5,2],
     [1,3,4,5,1],],
])
print(a.shape)
l=(1,2)
print((a[l][:]))
a[l]=0
# print(np.max(a[:,:,1]))
# print(np.max(a[:,:,2]))
# print(np.max(a[:,:,3]))
# print(np.max(a[:,:,4]))

# import itertools
# for i, j ,k in itertools.product(range(5), range(5), range(5)):
#     print(i,j,k)

# for i, j in range(5),range(5):
#     print(i,j)

# arr=np.array([1,2,4,5,6])
# print(arr[-1], arr[-2])