# -*-coding:utf-8 -*-


#import CutFile_ZH as cutfile
#import gensim_lda
import pymysql
import itertools
from preocr import OcrText

hostname = "127.0.0.1"
port = 3306
username = "root"
pwd = "123456"
database = "device"


def readRandSku(typeid,count):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select distinct `item_sku_id` from jd_item_rec where item_third_cate_id='"+str(typeid)+"' ORDER BY RAND() LIMIT "+str(count))

    skulist = list(itertools.chain.from_iterable(cur))

    cur.close()
    conn.close()
    return skulist

def readComments(sku):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select content from jd_item_comment where item_sku_id ="+str(sku))
    
    commentlist = list(itertools.chain.from_iterable(cur))
    
    cur.close()
    conn.close()
    return commentlist


def readOcrs(sku):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select distinct ocr_text from jd_item_ocr o,jd_item_rec r  where o.item_id=r.item_id and r.item_sku_id = "+str(sku))
    
    resultList =list(itertools.chain.from_iterable(cur))

    cur.close()
    conn.close()
    return resultList

def readAllOcrs():
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select ocr_text from jd_item_ocr")
    
    resultList =list(itertools.chain.from_iterable(cur))
    
    cur.close()
    conn.close()
    return resultList

def read_one_recommend(sku):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select ocr_text from jd_item_ocr where item_id = "+str(sku))
    
    resultList =list(itertools.chain.from_iterable(cur))
    
    cur.close()
    conn.close()
    return resultList



def read_type_recommends(typeid):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select recommend_reason from jd_item_rec where item_third_cate_id = "+str(typeid))
    
    resultList =list(itertools.chain.from_iterable(cur))
    
    cur.close()
    conn.close()
    return resultList

def read_sku_title_rectitle(sku):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select sku_name,recommend_theme,recommend_reason from jd_item_rec where item_sku_id = "+str(sku))
    result = cur.fetchone()
#    resultStr = ""
#    if result == None:
#        resultStr = ""
#    else:
#        resultStr = result[0]
    cur.close()
    conn.close()
    return result

def read_sku_pty(sku):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select value_name from jd_item_pty where item_sku_id = "+str(sku))
    
    resultList =list(itertools.chain.from_iterable(cur))
    
    cur.close()
    conn.close()
    return resultList


def read_sku_pty_key(sku,key_name):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select value_name from jd_item_pty where item_sku_id = "+str(sku)+ " and key_name='"+str(key_name)+"'")
    
    result = cur.fetchone()

    cur.close()
    conn.close()
    return result


def read_sku_brand(sku):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
#    cur.execute("select value_name from jd_item_pty where item_sku_id = "+str(sku)+ " and key_name='品牌'")
    cur.execute("select brandname_full from jd_item_brand where item_sku_id = "+str(sku))

    result = cur.fetchone()
    
    cur.close()
    conn.close()
    return result

def read_distinct_brand(sku=None):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    if sku == None:
        cur.execute("select distinct brandname_en,brandname_cn from jd_item_brand ")
    else:
        cur.execute("select distinct brandname_en,brandname_cn from jd_item_brand where item_sku_id = "+str(sku))
    
    resultList =list(itertools.chain.from_iterable(cur))

    cur.close()
    conn.close()
    return resultList

def read_similiar_recommends(feature,typeid):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select recommend_reason from jd_item_rec where item_third_cate_id = "+str(typeid) + " and recommend_reason like '%"+feature+"%'")
    
    resultList =list(itertools.chain.from_iterable(cur))
    
    cur.close()
    conn.close()
    return resultList



def read_pry_typeid(skuid,typeid,choosed_pty_list):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    sql = "select distinct value_name from jd_item_pty where CHAR_LENGTH(value_name)>1 and value_name  NOT REGEXP '^[-+]?[0-9]*\.?[0-9]+$' and  item_third_cate_id = "+str(typeid)+" and item_sku_id <> '"+str(skuid)+"' "
    if choosed_pty_list!=None:
        pty_string = ("','".join(choosed_pty_list))
        sql += " and key_name in ('"+pty_string+"')"
    cur.execute(sql)
    
    
    resultList =list(itertools.chain.from_iterable(cur))
    
    cur.close()
    conn.close()
    return resultList

def read_raw_ocr(sku=None):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    if sku == None:
        cur.execute("select distinct item_id, imgurl,ocr_raw_result from jd_item_raw_ocr where ocr_raw_result <>'FAIL' order by item_id")
    else:
        cur.execute("select distinct item_id, imgurl,ocr_raw_result from jd_item_raw_ocr where ocr_raw_result <>'FAIL' and item_id = "+str(sku))

    resultList = cur.fetchall()

    cur.close()
    conn.close()
    return resultList

def save_img_item(item_id,imgurl,result_text,priority):
#    print(result_text)
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("""
       REPLACE INTO jd_item_ocr
        (item_id, imgurl, ocr_text,priority)
        VALUES
        (%s, %s, %s,%s)
        """, (item_id, imgurl, result_text,priority)     # python variables
    )
    conn.commit()

    cur.close()
    conn.close()
    return True

def save_user_dict(user_dict):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    for d in user_dict.keys():
        cur.execute("""
            REPLACE INTO jd_user_dict
            (words, sim, frequency)
            VALUES
            (%s, %s, %s)
            """, (d, 0, user_dict[d])     # python variables
            )
    conn.commit()
    
    cur.close()
    conn.close()
    return True

def read_user_word(word):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select * from jd_user_dict where words ='"+str(word)+"'")
    result = False
    exist = cur.fetchone()
    if exist:
        result = True
    
    cur.close()
    conn.close()
    return result

def read_dict_word(word):
    conn = pymysql.connect(host=hostname, port=int(port), user=username, passwd=pwd,db=database, unix_socket="/tmp/mysql.sock",charset='utf8')
    cur = conn.cursor()
    cur.execute("select word_count from wiki_two_word where word ='"+str(word)+"'")
    result = 0
    exist = cur.fetchone()
    if exist:
        result = exist[0]
    
    cur.close()
    conn.close()
    return result
