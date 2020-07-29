import requests
import shutil
import numpy as np

DataSet = "BX-Books.csv"

links =  np.genfromtxt(DataSet, delimiter=";", usecols=(0), loose=False, invalid_raise=False, dtype=str)

for i in range(104315, len(links)):
    r = requests.get(links[i], stream = True)
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(str(i) + ".jpg",'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully Downloaded: ',i)
    else:
        print('Image Couldn\'t be retreived')


