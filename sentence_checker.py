#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
import multiprocessing
import gensim  
from gensim.models import Word2Vec
import jieba
import util_tool as util
import re
import mysql_reader as db
from aip import AipNlp
import math

""" 你的 APPID AK SK """
APP_ID = '10354266'
API_KEY = 'gdbawfxx8rQoWXGezBxFhXR6'
SECRET_KEY = 'aP8Zk1TNmIhYyf6Lwu1SXjsgeq2oSm2L'


class SentenceChecker:
    def __init__(self,model_path=None,user_dict=None):
        print("initing SentenceChecker...")
        if model_path == None:
            model_path = "../wordvec/model/zhwiki"
        if user_dict == None:
            user_dict = "all_brand.txt"
        jieba.load_userdict(user_dict)
        print("loading word2vec model...")
        self.model = Word2Vec.load(model_path)
        self.aipNlp = AipNlp(APP_ID, API_KEY, SECRET_KEY)


    def add_user_words(self,words_list):
        for word in words_list:
            if word != None and len(word)>0:
                jieba.add_word(word)

    def word2vec_eval(self, word):
        if word in self.model.wv.vocab:
            return True
        else:
            return False

    def get_vector(self,word):
        return model.wv[str(word)]

    def remove_punc(self,line_sentence):
        multi_version = re.compile("-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-")
        punctuation = re.compile("[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")
        line = multi_version.sub(r"\2", line_sentence)
        line = punctuation.sub('', line_sentence)
        return line
    
    
    
    def score_sentence(self,sentence):
        if len(sentence.strip())<=1:
            return 100000000
        
        result = self.aipNlp.dnnlm(sentence)
        ppl = result['ppl']
        return ppl
        
    def is_good_phase(self,sentence):
        result = self.aipNlp.dnnlm(sentence)
        print(result)

        ppl = result['ppl']
        
        if ppl < 6000:
            return True
        else:
            print("####"+sentence+" err:",ppl)
        
        return False
    
    def check_sentence(self,sentence):
        if sentence == None or len(sentence)<4:
            return False
        
        is_good = self.is_good_phase(sentence)
        if is_good == False:
            return False
        
#        line = self.remove_punc(sentence).strip()
        line = sentence.strip()
        count = 0
        errword = ""
        words = jieba.cut(line)
        for word in words:
#            result = self.word2vec_eval(word)
#            print(word)
            if len(word)<2:
                continue
            if word.isdigit():
                continue
            if re.match("^[A-Za-z0-9.]*$", word):
                continue

            result = db.read_user_word(word)
            
            if result==False and word not in errword:
                count += 1
                errword = errword + " " + word
        if len(sentence)>8 and count > 1:
            print(line+"   #####异常词： "+errword)
            return False
        elif count>0:
            return False
        else:
            return True

    #RETURN percent of err words like 95%,only return 95
    def score_sentence_dict(self,sentence):
        if sentence == None or len(sentence)<1:
            return 0
            
        #        line = self.remove_punc(sentence).strip()
        line = sentence.strip()
        words = []
        skip_next = False
        wrong_words_count = 0
        for i,ch in  enumerate(line):
            if skip_next:
                skip_next = False
                continue
            if util.isChinese(ch) and util.isChinese(line[i+1]):
                words.append(ch+line[i+1])
                skip_next = True
        #如果有*等特殊符号，则认为存在识别错误，对本句进行惩罚
        if "*" in line:
            wrong_words_count = 50

        total_count =0
        count = 0
        errword = ""
        for word in words:
            if len(word.strip())<=0:
                continue
            
            result = db.read_dict_word(word)
            
            if result<=1 and word not in errword:
                count += 1
                errword = errword + " " + word
                print("@@@@@@@@@@@@@",errword)
            total_count +=1
                
        if total_count>0:
            result_socre = (count*10)*100/total_count + wrong_words_count
            return result_socre
        else:
            return 100
        

    def eval_sentence(self,sentence):
        #TODO EVAL SENTENCE SCORE
        #a. DL model
        #b. Dict
        #c. GENSIM SIMILITY
        #score = a*x + b*y + c*z ,x=0.3,y=0.4,z=0.3
        #分值越大越差
        dl_score = self.score_sentence(sentence)
        print("dl score:",dl_score)
#        dl_score = math.log(dl_score) if dl_score>0 else 0
        dl_score = (dl_score/1000)**2

        print("log dl score:",dl_score)

        dict_score = self.score_sentence_dict(sentence)
        print("dict score:",dict_score)

        score = dl_score + dict_score
        return score

    def eval_phase(self,phase):
        #TODO EVAL SENTENCE SCORE
        #a. DL model
        #b. Dict
        #c. GENSIM SIMILITY
        #score = a*x + b*y + c*z ,x=0.3,y=0.4,z=0.3
        #分值越大越差
        
        dl_score = self.score_sentence(phase.phase_string())
        print("dl score:",dl_score)
#        dl_score = math.log(dl_score) if dl_score>0 else 0
        dl_score = dl_score/1000
        print("log dl score:",dl_score)
        avg_length = sum([len(s.sentence) for s in phase.sentence_list])/len(phase.sentence_list)
        #平均字数在10字左右的句子质量较高
        distance = abs(avg_length-10)
        
        score = dl_score +  distance
        return score

if __name__ == '__main__':
    cc = SentenceChecker()
    cc.check_sentence("土阅读内窨来自荣耀阅读")


