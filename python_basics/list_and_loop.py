'''
Write a Python function that takes a list of integers as
input and returns the sum of all even numbers in the list.
'''

numbers = []

n = int(input("Enter the number of items in the list : "))

for i in range(n) : 
    x = int(input(f"Enter number {i+1} : "))
    numbers.append(x)


even_numbers = []
for item in numbers[0:n]:
    if (item % 2 ==0):
        even_numbers.append(item)

sum = 0
for item in even_numbers[0:n]:
    sum = sum + item


print(even_numbers, sum)