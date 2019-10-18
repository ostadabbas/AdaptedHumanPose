'''
Fundamental utils.
'''
import re
import os
import sys

def getNumInStr(str_in, tp=int):
	'''
	get the number in list transferred as type(indicated)
	:param str_in: the input string
	:return:
	'''
	temp = re.findall(r'\d+', str_in)
	res = list(map(tp, temp))
	return res

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

