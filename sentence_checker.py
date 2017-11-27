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
from stanfordcorenlp import StanfordCoreNLP

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
        self.nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2017-06-09/', lang='zh')


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
        ppl = 10000
        if 'ppl' in result:
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
        for sp in ["*",":",")","(","/"]:
            if sp in line:
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
            result_socre = (count)*100/total_count + wrong_words_count
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
        print("sentence",sentence)
        dl_score = self.score_sentence(sentence)
        print("dl score:",dl_score)
#        dl_score = math.log(dl_score) if dl_score>0 else 0
        dl_score = (dl_score/600)

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
        good_sentence_length = 10
        print("phase",phase.phase_string())
        dl_score = self.score_sentence(phase.phase_string())
        print("dl score:",dl_score)
#        dl_score = math.log(dl_score)*5 if dl_score>0 else 0
        dl_score = (dl_score/600)
        print("log dl score:",dl_score)
        avg_length = sum([len(s.sentence) for s in phase.sentence_list])/len(phase.sentence_list)
        #平均字数在10字左右的句子质量较高
        sub_words_count = abs(avg_length - good_sentence_length)
        print(sub_words_count)
        if sub_words_count ==0:
            distance = 0
        else:
#            distance = math.log(sub_words_count)*5
            distance = (sub_words_count)*5

        #子句数量太多，惩罚
        if len(phase.sentence_list)>2:
            distance += (len(phase.sentence_list)-2)*2
        np_percent_score = 0
        
        #只有一个子句，且子句过短，惩罚
        if len(phase.sentence_list)==1:
            mm = abs(20 - len(phase.sentence_list[0].sentence))
            distance += (mm*5)
        
            #判断句子中没有谓语
            parser_result = self.nlp.parse(phase.phase_string())
            if "VP" not in parser_result and "NP" in parser_result:
                distance +=50
        else:
            #处理多个小短句中NP很多的情况，惩罚罗列NP
            np_count = 0
            for s in phase.sentence_list:
                print(s)
                sentence_result = self.nlp.parse(s.sentence)
                if "VP" not in sentence_result and "NP" in sentence_result:
                    np_count += 1
            np_percent = 10*np_count/len(phase.sentence_list)
            print("惩罚罗列NP",np_percent)
            np_percent_score = np_percent
        
        #征罚人为断句
        user_cut_score = 0
        user_cut_count = 0
        for s in phase.sentence_list:
            if s.user_cut:
                user_cut_count+=1
        user_cut_score = 10*user_cut_count/len(phase.sentence_list)
        
        score = dl_score +  distance + np_percent_score + user_cut_score
        return score

if __name__ == '__main__':
    cc = SentenceChecker()
    cc.check_sentence("土阅读内窨来自荣耀阅读")


