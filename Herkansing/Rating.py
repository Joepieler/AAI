import numpy as np

BooksFile = "BX-Books.csv"
BooksRating = "BX-Book-Ratings.csv"

#books_numbers = np.genfromtxt(BooksFile, delimiter=";", usecols=(0), loose=False, invalid_raise=False, dtype=str)
Books_Rating = np.genfromtxt(BooksRating, delimiter=";", usecols=(1,2), loose=False, invalid_raise=False, dtype=str)

dict_books = {}
for i in Books_Rating:
    if i[1] != "0":
        if i[0] in dict_books:
            rating = dict_books[i[0]]
            rating.append(int(i[1]))
            dict_books[i[0]] = rating
        else:
            dict_books[i[0]] = [int(i[1])]

for key in dict_books:
    rating = dict_books[key]
    dict_books[key] = sum(rating) / len(rating)

print(dict_books)
