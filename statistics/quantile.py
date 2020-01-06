#coding:utf-8
import math

def quantile_p(data, p):
	pos = (len(data) + 1) * p
	pos_integer = int(math.modf(pos)[1])
	pos_decimal = pos - pos_integer
	Q = data[pos_integer - 1] + (data[pos_integer] - data[pos_integer - 1]) * pos_decimal
	return Q

if __name__ == "__main__":
	data = [6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49]
	Q1 = quantile_p(data, 0.25)
	print("Q1:", Q1)
	Q2 = quantile_p(data, 0.5)
	print("Q2:", Q2)
	Q3 = quantile_p(data, 0.75)
	print("Q3:", Q3)
