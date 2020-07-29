import numpy as np
import csv

BooksFile = "BX-Books-full.csv"
BooksRating = "BX-Book-Ratings.csv"

Books_Numbers = np.genfromtxt(BooksFile, delimiter=";", usecols=(0), loose=False, invalid_raise=False, dtype=str)
Books_Rating = np.genfromtxt(BooksRating, delimiter=";", usecols=(1,2), loose=False, invalid_raise=False, dtype=str)
'''
Get a list of ratings per book
'''
def GetBookRatings(Books_Rating):
    dict_books = {}
    for i in Books_Rating:
        if i[1] != "0":
            if i[0] in dict_books:
                rating = dict_books[i[0]]
                rating.append(int(i[1]))
                dict_books[i[0]] = rating
            else:
                dict_books[i[0]] = [int(i[1])]
    return dict_books


'''
Get Rating from bad medium good
'''
def GetRating(dict_books):
    dict_books_labels = {}
    for key in dict_books:
        rating = dict_books[key]
        avr_rating = sum(rating) / len(rating)
        if avr_rating >= 1 and avr_rating <= 4:
            dict_books_labels[key] = "bad"
        elif avr_rating >= 5 and avr_rating <= 7:
            dict_books_labels[key] = "medium"
        elif avr_rating >= 8 and avr_rating <= 10:
            dict_books_labels[key] = "good"
    return dict_books_labels


'''
get only the books that are ratted bad or good
'''
def GetBadGoodRatings(dict_books):
    bad_good_books = {}
    for key in dict_books:
        if dict_books[key] == "bad" or dict_books[key] == "good":
            bad_good_books[key] = dict_books[key]
    return bad_good_books

def GetCoverdRatings(Books_Rating, books_numbers):
    books_ratings = GetRating(GetBookRatings(Books_Rating))
    books_ratings_covers = {}
    for i in range(0,121764):
        if books_ratings.get(books_numbers[i]):
            books_ratings_covers[books_numbers[i]] = books_ratings.get(books_numbers[i])
        else:
           books_ratings_covers[books_numbers[i]] = "Null"


    return books_ratings_covers



books_ratings = GetCoverdRatings(Books_Rating, Books_Numbers)


w = csv.writer(open("output.csv", "w"))
for key, val in books_ratings.items():
    w.writerow([key, val])



