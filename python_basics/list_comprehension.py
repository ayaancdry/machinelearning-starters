'''
Given a list of words as input, create a new list
containing the length of each word using list comprehension.
'''
words = input("Enter a list of words: ").split()
lengths = [len(word) for word in words]
print("List of word lengths:", lengths)

