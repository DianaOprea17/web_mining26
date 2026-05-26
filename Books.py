# Exercițiul 1: Extrageți titlul, prețul și disponibilitatea cărților de pe pagina principală
# a site-ului http://books.toscrape.com. Normalizați prețurile (float) și disponibilitatea
# (1 pentru "In stock", 0 altfel), apoi stocați-le într-un DataFrame pandas.

# ex3 - c1

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://books.toscrape.com/"

response = requests.get(url)

# === START ===
soup = BeautifulSoup(response.text, 'html.parser')
books = soup.find_all('article', class_='product_pod')

print("Scraped books:\n")

book_list = []

for book in books:
    # title
    title = book.find("h3").find("a")["title"]

    # extract + normalize price
    price_text = book.find("p", class_="price_color").text.strip()
    price_text = price_text.encode("ascii", "ignore").decode()

    # availability
    availability_text = book.find("p", class_="instock availability").text.strip()

    if "In stock" in availability_text:
        availability = 1
    else:
        availability = 0

    print("Title: " + title)
    print("Price: " + price_text)
    print("Availability: " + availability_text)
    print("\n")

    book_list.append([title, price_text, availability])

df = pd.DataFrame(book_list)

print("\nDataFrame:\n")
print(df)

