import mysql.connector as msc
# import pymysql.cursors as msc

def connect():
    _cnx = msc.connect(user='root', password='C0mic$toR3',
                       host='52.14.48.229',
                       database='comicstore')
    _cursor = _cnx.cursor()
    return _cnx, _cursor


def allComics():
    query = ("SELECT distinct (image_link) image_link "
             "FROM comicstore.diamond_comics_source "
             "WHERE image_link != '' "
             "AND category in (1,2,3,4) "
             "ORDER BY id limit 5000")

    cursor.execute(query)
    urls = []
    for (image_link) in cursor:
        urls.append("http://comicstore.cf%s" % image_link)

    return urls

def executeQuery():
    query = ("SELECT image_link FROM comicstore.diamond_comics_source WHERE image_link != ''"
             # "AND full_title like '%BATMAN%'"
             "AND full_title like '%SUPERMAN%' "
             "AND category in (1,2,3,4)"
             "ORDER BY id LIMIT 1000")

    cursor.execute(query)

    urls = []
    for (image_link) in cursor:
        urls.append("http://comicstore.cf%s" % image_link)

    return urls


def close():
    cursor.close()
    cnx.close()


cnx, cursor = connect()
