import mysql.connector as msc


def connect():
    cnx = msc.connect(user='root', password='C0mic$toR3',
                      host='52.14.48.229',
                      database='comicstore')

    cnx.close()
