
#LAB 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.isomorphism.matchhelpers import numerical_doc
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from sympy import rotations

#NUMPY OPn
arr1d = np.array([1, 2, 3, 4])
print("1D array:", arr1d)

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("1D array:", arr2d)

print("1D Array sum: ", arr1d.sum())
print("Mean: ", arr1d.mean())
print("Std Deviation: ", arr1d.std())

reshaped_arr = arr1d.reshape(2, 2)
print("1D Array reshaped: ", reshaped_arr)

reshaped_arr = arr2d.reshape(3, 2)
print("2D Array reshaped: ", reshaped_arr)


#PANDAS OPn
data = {
    'Name': ['John', 'Rock', 'Will', 'Smith'],
    'Age': [28, 22, 32, 37],
    'Salary': [50000, 45000, 55000, 65000],
    'Department': ['IT', 'HR', 'IT', 'Finance']
}

df = pd.DataFrame(data)
print("Sample Data:\n", df)

print("Dataframe info:")
print(df.info())

print("Dataframe description:")
print(df.describe())


#MATPLOTLIB
#Line Plot
plt.subplot(1, 1, 1)
x = np.linspace(0, 10, 50)
y = np.sin(x)
print('X values:', x)
print('Y values:', y)
plt.plot(x, y)
plt.title('Line Plot: Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
#plt.show()

#Scatter Plot
plt.subplot(1, 1, 1)
x = np.random.normal(0, 1, 50)
y = np.random.normal(0, 1, 50)
print('X values:', x)
print('Y values:', y)
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

#Histogram
plt.subplot(1, 1, 1)
data = np.random.normal(0, 1, 1000)
print('data values:', data)
plt.hist(data, 30)
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
#plt.show()


#Box Plot
plt.subplot(2, 2, 4)
sns.boxplot(x='Department', y='Salary', data=df)
plt.title('Salary Distribution by department')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation HeatMap')
plt.tight_layout()
plt.show()


#Tensors
#creating tensor from numpy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor_from_numpy =  torch.from_numpy(numpy_array)
print('Tensor from numpy:\n', tensor_from_numpy)

#creating tensor directly
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('Direct Tensor:\n', tensor1)

tensor2 = torch.zeros([2, 3])
tensor3 = torch.ones([2, 3])
tensor4 = torch.randn([2, 3])
tensor5 = torch.arange(8, 10, 2)
print('Zeroes Tensor:\n', tensor2)
print('Ones Tensor:\n', tensor3)
print('Random Tensor:\n', tensor4)
print('Arange Tensor:\n', tensor5)


#Tensor OPn
print('\nTensor Operations:')
print('Sum:', tensor1.sum())
print('Mean:', tensor1.float().mean())
print('Addition:\n', tensor1 + tensor1)
print('Multiplication:\n', tensor1 * tensor1)
print('Matrix Multiplication:\n', torch.mm(tensor1, tensor1.T))

print('Reshape:', tensor1.reshape(3, 2))
print('Transpose:', tensor1.T)
print('Concatenate:', torch.cat([tensor1, tensor1], dim=0))
print('Stack:', torch.stack([tensor1, tensor1]))




