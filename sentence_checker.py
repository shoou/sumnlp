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
from snownlp import SnowNLP


""" 你的 APPID AK SK """
#APP_ID = '10354266'
#API_KEY = 'gdbawfxx8rQoWXGezBxFhXR6'
#SECRET_KEY = 'aP8Zk1TNmIhYyf6Lwu1SXjsgeq2oSm2L'


#APP_ID ='10452447'
#API_KEY = 'v4x9cYYb1cdIj77vXQiV2obo'
#SECRET_KEY = 'QdgaeR8FmGI2ZvCiqL3IS50WOQwvhG2g'

APP_ID ='10461903'
API_KEY = '4S3cRWw5Gb8zpVORqhTWGb4M'
SECRET_KEY = 'S6hVNuwxzZopUu28lS9jpGAYIMbayPoE'



class SentenceChecker:
    def __init__(self,model_path=None,user_dict=None):
        print("initing SentenceChecker...")
        if model_path == None:
            model_path = "../../wordvec/model/zhwiki"
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
    
    
    
    def score_sentence2(self,sentence):
        score_result = 100000
        if len(sentence.strip())<=1:
            return score_result
        seg_words = self.nlp.word_tokenize(sentence)
        print(seg_words)
        words = [w for w in seg_words if w not in [',','，','。','.','!','！','?','？','%','%',';','；',':','：']]
#        words = jieba.cut(sentence)
        word_count = 0
        avg_sim = 0
        for idx,word in enumerate( words):
            low_sim = 10000
            word_count+=1
            if word not in self.model.wv.vocab:
                print(word +" not in vocab")
                continue
            for idx2,word2 in enumerate(words):
                if idx2 > idx:
                    if word2 not in self.model.wv.vocab:
                        print(word2+" not in vocab two")
                        continue
                    sim = self.model.similarity(word,word2)
                    if sim < low_sim:
                        low_sim = sim
            if low_sim != 10000:
                avg_sim += low_sim
            
        avg_sim = avg_sim/word_count
        print(avg_sim)
        return avg_sim
    
    
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
            #不检测英文单词,这里要用英文词典，补上英文检测
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
            result_socre = (count)*5/total_count + wrong_words_count
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
#        good_sentence_length = 10
#        sentence_length = len(sentence)
#        sub_words_count = abs(sentence_length - good_sentence_length)

        print("log dl score:",dl_score)

        dict_score = self.score_sentence_dict(sentence)
        print("dict score:",dict_score)



        score = dl_score + dict_score
        return score
    #分值越大越差
    def eval_phase(self,phase):
        
        phase_string =phase.phase_string()
        good_sentence_length = 8
        print("phase",phase_string)
        dl_score = self.score_sentence(phase_string)
        print("dl score:",dl_score)
#        dl_score = math.log(dl_score)*5 if dl_score>0 else 0
        dl_score = (dl_score/600)
        print("log dl score:",dl_score)
        avg_length = sum([len(s.sentence) for s in phase.sentence_list])/len(phase.sentence_list)
        #平均字数在8字左右的句子质量较高
        sub_words_count = abs(avg_length - good_sentence_length)
        print(sub_words_count)
        if sub_words_count ==0:
            distance = 0
        else:
#            distance = math.log(sub_words_count)*5
            distance = (sub_words_count)*2

        #子句数量太多，惩罚
        if len(phase.sentence_list)>2:
            distance += (len(phase.sentence_list)-2)*3
        np_percent_score = 0
        
        #只有一个子句，且子句过短，惩罚
        if len(phase.sentence_list)==1:
            mm = abs(20 - len(phase.sentence_list[0].sentence))
            distance += (mm*5)
        
            #判断句子中没有谓语
            parser_result = self.nlp.parse(phase_string)
            if "VP" not in parser_result and "NP" in parser_result:
                distance +=50
        else:
            #处理多个小短句中NP很多的情况，惩罚罗列NP
            np_count = 0
            for s in phase.sentence_list:
                sentence_result = self.nlp.parse(s.sentence)
#                print(s,sentence_result)
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
        user_cut_score = 20*user_cut_count/len(phase.sentence_list)

        err_structure_socre = 0
        #以‘的’‘得’‘地’打头的句子惩罚
        if phase_string[0] in ['的','得','地']:
            err_structure_socre += 20
        if phase_string[0] in ['我','但','更']:
            err_structure_socre +=5

        has_special_char = False
        special_char_score = 0

        for punc in ['?','？','<','>','[',']','【','】']:
            if punc in phase_string:
                has_special_char = True
                special_char_score += 5

        sentiment_score = SnowNLP(phase_string).sentiments*10
        score = dl_score +  distance + np_percent_score + user_cut_score + special_char_score
        print("sentiment_score",score,sentiment_score)

        return score

if __name__ == '__main__':
    cc = SentenceChecker()
    cc.score_sentence2("土阅读内窨来自荣耀阅读")
    cc.score_sentence2("还支持像素级动态对比度调整技术")
    cc.score_sentence2("既能在阳光下清洗阅读")
    cc.score_sentence2("前置1600万摄像头，美颜自拍让你的笑容迷人F1.9大光圈支持夜间拍摄")
    cc.score_sentence2("拥有人工智能的美图M8，是你的拍照机器人。人工智能实现背景识别与美化，能够分辨照片中的人像和背景，为你带来出色的背景美化")



