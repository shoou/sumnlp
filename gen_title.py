#!/usr/bin/python
#coding:utf-8
import sys
import jieba
import jieba.analyse
import numpy as np

#from gensim import corpora, models
import jieba.posseg as pseg
import time
from pathlib import Path
#import gensim
#from gensim.models import Word2Vec
import re
import mysql_reader as db
import util_tool as util
#import thulac
from gen_content import ContentCreater


#是否显示调试信息
global in_debug
in_debug = False


class TitleCreater:

    def __init__(self):
        jieba.load_userdict("all_brand.txt")
        stop_word_file ="./title_stopwords.txt"
        jieba.analyse.set_stop_words(stop_word_file)
        self.stop_words_set = self.get_stop_words_set(stop_word_file)
        if in_debug:print("共计导入 %d 个停用词"%len(self.stop_words_set))
        self.contentCreater = ContentCreater()

    def remove_brackets(self,title_sentence):
        ret = ''
        skip1c = 0
        skip2c = 0
        skip3c = 0
        skip4c = 0
        for i in title_sentence:
            if i == '[':
                skip1c += 1
            elif i == '(':
                skip2c += 1
            elif i == '【':
                skip3c += 1
            elif i == '（':
                skip4c +=1
            elif i == ']' and skip1c > 0:
                skip1c -= 1
            elif i == ')'and skip2c > 0:
                skip2c -= 1
            elif i == '】'and skip3c > 0:
                skip3c -= 1
            elif i == '）'and skip4c > 0:
                skip4c -= 1

            elif skip1c == 0 and skip2c == 0 and skip3c == 0 and skip4c==0:
                ret += i
        return ret


    def remove_punc(self,line_sentence):
        multi_version = re.compile("-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-")
        punctuation = re.compile("[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")
        line = multi_version.sub(r"\2", line_sentence)
        line = punctuation.sub('', line_sentence)
        return line


    def cut_words(self,sentence):
        word_list =[]
        words = jieba.cut(sentence)
        for word in words:
            if word in self.stop_words_set:
                if in_debug:
                    print("ignore words:"+word)
                continue
            if len(word.strip())>1:
                word_list.append(word)

        return word_list

    #def cut_words_feature(sentence,filter):
    #    word_list =[]
    #    words = pseg.cut(sentence)
    #    print("pseg.cut标题分词:")
    #    for word,flag in words:
    #        print(word+" "+flag,end=",")
    #        if (flag in filter) and len(word.strip())>1:
    #            word_list.append(word)
    #
    #    print("\n")
    #    print("pseg.cut 过滤 "+str(filter)+"之后分词："+str(word_list))
    #    return word_list

    def cut_words_feature(self,sentence,filter):
        word_list =[]
        words = jieba.cut(sentence)
        for word in words:
            flags = pseg.cut(word,HMM=True)
            for x,flag in flags:
                if in_debug:
                    print(x+" "+flag,end=",")
                if (x in self.stop_words_set):
                    if in_debug:
                        print("ignore words:"+x)
                    continue
                if (flag in filter) and len(word.strip())>1:
                    word_list.append(word)
        if in_debug: print("\npseg.cut 过滤 "+str(filter)+"之后分词："+str(word_list))
        return word_list

    #def cut_words(sentence):
    #    word_list =[]
    #    words = thu1.cut(sentence, text=False)
    #    for word in words:
    #        word = word[0]
    #        if len(word.strip())>1:
    #            word_list.append(word)
    #
    #    return word_list
    #
    #def cut_words_feature(sentence,filter):
    #    word_list =[]
    #    words = thu1.cut(sentence, text=False)
    #    for word in words:
    #        flag = word[1]
    #        w = word[0]
    #        print(w+" "+flag,end=",")
    #        if (flag in filter) and len(w.strip())>1:
    #            word_list.append(w)
    #
    #    print("\npseg.cut 过滤 "+str(filter)+"之后分词："+str(word_list))
    #    return word_list



    def cut_sku_words(self,in_lines):
        out_lines = []
        for line in in_lines:
            line = self.remove_punc(line)
            line = line.strip()
            if len(line) < 1:  # empty line
                continue
            line_words =""
    #        for word in jieba.cut(line):
            for word,x in jieba.analyse.textrank(line.strip(), withWeight=True,allowPOS=('n', 'vn')):
                if word.strip()=="":
                    continue
                line_words = line_words + (word + " ")
            out_lines.append(line_words)
        return out_lines

    def textrank_words(self,new_title,filter,features_count,type_name):
        wordlist = []
        textrank_words = jieba.analyse.textrank(new_title.strip(), withWeight=True,allowPOS=filter)
    #    textrank_words = thu1.cut(new_title.strip(), text=False)
        if in_debug:
            print(str(filter)+"分词" + str(textrank_words))
        for words in textrank_words:
            w = words[0]
            flag = words[1]
            if flag not in filter:
                continue
            if w == type_name:
                continue
            features_count = features_count-1
            if features_count <0:
                break
            wordlist.append(w)
        
        return wordlist



    def get_stop_words_set(self,file_name):
        with open(file_name,'r') as file:
            return set([line.strip() for line in file])


    def get_brand(self,skuid):
        brand = ""
        brandRow = db.read_sku_brand(skuid)
        if brandRow != None:
            brand = self.remove_brackets(brandRow[0])
        
        return brand

    def get_model(self,skuid):
        model = ""
        modelRow = db.read_sku_pty_key(skuid,"型号")
        if modelRow != None:
            model = self.remove_brackets(modelRow[0])
        model_parts = model.split(" ")
        if len(model_parts)>3:
            model = " ".join(model_parts[:3])

    #    if len(model)<1:
    #        modelRow = db.read_sku_pty_key(skuid,"品牌")
    #        if modelRow != None:
    #            model = remove_brackets(modelRow[0])
        return model

    def create_title(self,type_id,type_name,sku_count,sku_ids):
        
        if sku_ids != None:
            skulist = sku_ids
        else:
            skulist = db.readRandSku(str(type_id),int(sku_count))
        result_list = []
        
        for skuid in skulist:
            print("processing sku id:"+str(skuid))
            recommend_reason = ""
            titleRow = db.read_sku_title_rectitle(skuid)
            if titleRow == None:
                continue
            title = titleRow[0]
            recommend_title = titleRow[1]
            recommend_reason = titleRow[2]
        
            if len(title)<3:
                print("title should more than 3 words:"+title)
                continue
            #生成品牌和型号
            brand = self.get_brand(skuid)
            model = self.get_model(skuid)
            brand_model = ""
            if len(brand)>1:
                if brand in model:
                    brand_model = model
                else:
                    if len(model)>1:
                        brand_model = brand +" "+ model
                    else:
                        brand_model = brand
            else:
                brand_model = model

            title = self.remove_brackets(title)

            new_title = self.remove_punc(title).strip()
            print("title:"+title)
            new_title = new_title.replace(brand,"")
            new_title = new_title.replace(model,"")
            if in_debug:
                print("去掉【"+brand+"】和【"+model+"】："+new_title)
            words = self.cut_words(new_title)
            if in_debug:print("标题分词："+str(words))
            #cut方式分词
            features = self.cut_words_feature(new_title,['vn','v','a','an'])
            if type_name in features:
                features.remove(type_name)

            wordlist = self.textrank_words(new_title,['vn','v','a','an'],2,type_name)

            #合并cut和textrank的分词（大部分应该是一样的）
            features.extend(wordlist)
            if in_debug:print("合并cut和textrank的分词"+str(features))
            tmplist =[]
            for x in features:
                if x in tmplist:
                    continue
                tmplist.append(x)
            features = tmplist
            if in_debug:print("去重后特征："+str(features))
            #如果特征少于等于1个，从标题中选择一名词补充
            if len(features)<=1:
                cwordslist = self.cut_words_feature(new_title,['n','nr'])
                if len(cwordslist)>0:
                    cwordslist.extend(features)
                    features = cwordslist

                if type_name in features:
                    features.remove(type_name)

                wordlist = self.textrank_words(new_title,['n'],2,type_name)
                features.extend(wordlist)
                tmplist =[]
                for x in features:
                    if x in tmplist:
                        continue
                    tmplist.append(x)
                features = tmplist

            if len(features) > 2:
                features = features[:2]
            feature_string =  "".join(features)

            type_pty_name = db.read_sku_pty_key(skuid,"分类")
            if type_pty_name != None and len(type_pty_name)>0:
                tmp_type_name = type_pty_name[0]
                if tmp_type_name =="其它":
                    tmp_type_name = type_name
            else:
                tmp_type_name = type_name

            if tmp_type_name in feature_string:
                tmp_type_name = ""

            machine_title = title

            if len(title)<8:
                print("\n"+"="*16 + "机器标题:"+title)
            else:
                machine_title = brand_model + feature_string + tmp_type_name
                print("\n"+"="*16 + "机器标题:"+ machine_title)
            print("="*16 + "达人标题:"+recommend_title+"\n")

            content_features = self.cut_words_feature(new_title,['n','nr'])
            content_features = list(set(content_features))
            if type_name in content_features:
                content_features.remove(type_name)

            if len(content_features)>5:
                content_features = content_features[:5]
            mchine_content = self.contentCreater.gen_content(skuid,type_id,content_features,type_name,brand)

            print("达人文章:"+recommend_reason)
            print("机器内容:"+mchine_content)
            if len(mchine_content)>50:
#                result_list.append("skuid:"+str(skuid)+"\n达人文章:"+recommend_reason+"\n机器内容:"+mchine_content+"\n")
                result_list.append("skuid:"+str(skuid)+"\n"+machine_title+"\n"+mchine_content+"\n\n")

            util.writeList2File(type_id+".txt",result_list,'a')
            del result_list[:]

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 <type_id> <type_name> <count> type_id and type_name [655,9719,1392]#[手机，连衣裙，面膜]")
        sys.exit(1)
    type_id = sys.argv[1]
    type_name = sys.argv[2]
    sku_count = sys.argv[3]
    skulist = None
    if len(sys.argv)>4:
        skulist = sys.argv[4].split(",")
    titler = TitleCreater()

    titler.create_title(type_id,type_name,sku_count,skulist)
