''' 
Given two NumPy arrays as input, perform
element-wise multiplication and store the result in a new array.
'''
import numpy as np

list1 = []
list2 = []

n1 = int(input("Enter the number of elements in first array : "))
n2 = int(input("Enter the number of elements in second array : "))

if (n1 != n2):
    print("Element-wise Multiplication not possible")

else: 
    for item in range(n1):
        x = int(input("Enter the elements of first array : "))
        list1.append(x)

    for item in range(n2):
        x = int(input("Enter the elements of second array : "))
        list2.append(x)

array1 = np.array(list1)
array2 = np.array(list2)

list_new = []

for i in range(len(array1)):
    product = array1[i] * array2[i]
    list_new.append(product)

product_array = np.array(list_new)
print(product_array)


