#coding:utf-8
import numpy as np
import numpy.random as npr

def alias_setup(probs):
	'''
	probs: 某个概率分布
	返回: alias数组与prob数组
	'''
	k = len(probs)
	q = np.zeros(k) #对应prob数组
	j = np.zeros(k, dtype=np.int) #对应alias数组
	# sort the data into the outcomes with probabilities
	# that are larger and smaller than 1/K
	smaller = [] #存储比1小的列
	larger = [] #存储比1大的列
	for kk, prob in enumerate(probs):
		q[kk] = k*prob #概率
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)
	
	# Loop though and create little binary mixtures that
	# appropriately allocate the larger outcomes over the
	# overall uniform mixture.

	# 通过拼凑,将各个类别都凑为1
	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		j[small] = large #填充alias数组
		q[large] = q[large] - (1.0 - q[small]) #将大的分到小的上

		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return j, q

def alias_draw(j, q):
	'''
	输入: prob数组和alias数组
	输出: 一次采样结果
	'''
	k = len(j)
	# Draw from the overall uniform mixture.
	kk = int(np.floor(npr.rand() * K)) #随机取一列

    # Draw from the binary mixture, either keeping the 
    # small one, or choosing the associated larger one.
	if npr.rand() < q[kk]: #比较
	    return kk
	else:
		return j[kk]

k = 5
n = 100

# Get a random probability vector
probs = npr.dirichlet(np.ones(k), 1).ravel()

# construct the table
j, q = alias_setup(probs)

# generate variates
x = np.zeros(n)
for nn in xrange(n):
	x[nn] = alias_draw(j, q)




