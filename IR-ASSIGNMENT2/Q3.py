import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math



data = pd.read_csv('IR-assignment-2-data.txt', sep=" ", header=None)
rows = len(data.index)


display(data)


database_dict={}


for ind in data.index:
    if(data[1][ind]=='qid:4'):
        database_dict[ind]=data[0][ind]

temp =data.drop((data.index[len(database_dict):]))
np.savetxt('query4max.txt', temp.values, fmt='%s', delimiter=" ")
unsortedDb = list(database_dict.items()) 
database_dict = sorted(database_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)

def check(count,database_dict):
    for key, value in database_dict.items():
        if value == 0:
            count[0] += 1
        elif value == 1:
            count[1] += 1
        elif value == 2:
            count[2] += 1
        else:
            count[3] += 1

def calc(count):
    return math.factorial(count[3]) * math.factorial(count[2]) * math.factorial(count[1]) * math.factorial(count[0])
def findTotalFiles(database_dict):
    count = [0, 0, 0, 0]
    check(count,database_dict)
    return calc(count)

def run_findDCG(data,length):
    ans = data[0][1]
    for i in range(1, length):
        ans+= (data[i][1] / math.log2(i + 1))
    return ans

# %%
import math

def findDCG(data, length):
    return run_findDCG(data,length)


# %%
# database_dict contains sorted pairs
DCG_max_value = findDCG(database_dict, len(database_dict))
print("Max DCG: " ,DCG_max_value)

DCG_unsorted_value = findDCG(unsortedDb, len(unsortedDb))
print("nDCG whole Dataset: ",DCG_unsorted_value/DCG_max_value)

DCG_max_value_50 = findDCG(database_dict, 51)
DCG_unsorted_value_50 = findDCG(unsortedDb, 51)
print("nDCG at 50: ",DCG_unsorted_value_50/DCG_max_value_50) 

# %%
totalRelDocs = 44;
retrievedRelatedDocs= 0;
precision = []
recall = []

# %%
def plot_me(rec,prec):
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
def ff(TF_sorted):
    while i in range(len(TF_sorted)):
        if (TF_sorted[i][1][1] != 0):
            retrievedRelatedDocs= retrievedRelatedDocs+1
        precisionval.append(retrievedRelatedDocs/(i+1))
        recallval.append(retrievedRelatedDocs/totalRelDocs)
        i = i+1

def getPrecisionValueAndRecallValue(pair_values):
    TF_sorted = sorted(pair_values.items(), key=lambda x: (-x[1], x[0]))
    global totalRelDocs, retrievedRelatedDocs, precisionval, recallval
    
    ff(TF_sorted);
    
    plot_me(recallval,precisionval)


# %%


def toFloat(x):
    return float(x)


pair_values = {}
for ind in data.index:
    ddd, init, pac = 'qid:4', 1, 75
    if(ddd == data[init][ind]):
        s = toFloat(data.at[ind,init + pac][(init ** init) * 3:])
        temp = (s,data.at[ind,init - 1])
        pair_values[ind] = temp
        
getPrecisionValueAndRecallValue(pair_values)


