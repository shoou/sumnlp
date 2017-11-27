#!/usr/bin/python
#coding:utf-8
#'''
#   主要思路：
#   1. 对某一商品中的所有评论、ocr内容提取主题特征，选择名词、动名词、形容词、名形词作为特征关键词key，商品属性数据中的属性值作为特征候选项（字数少于2个的滤除)
#   1.1 使用句子依存分析，找出中心词作为特征词
#   2. 提取特征对应的观点，形成特征-观点对。
#   3. 协同过滤方法选择相似文章（目前达人文章数量不足，选择三级分类相同的所有达人文章）
#   4. 依次查找某一特征词对应的子句并验证合理性（规则条件过滤），同时，查找和验证其观点对应的子句
#   5. 所有子句中选择前5个特征作为子句，集成一段话。集成过程中进行连贯性重写（语义重写）。
#'''
import sys
import mysql_reader as db
import time
import util_tool as util
import jieba
import collections
import gensim
from gensim import corpora, models
from gensim.models import Word2Vec

import jieba.posseg as pseg
#from snownlp import SnowNLP
from sentence_checker import SentenceChecker
import re
import jnius_config
from jnius import autoclass
import os
import datetime
import math
from operator import itemgetter, attrgetter, methodcaller


#os.environ['CLASSPATH'] = '/Users/lully/Documents/MSE/JingDong/Code/CiLin/bin/cilin/cilin.jar'
#os.environ['CLASSPATH'] = '/Users/lully/Documents/MSE/JingDong/Code/demo/CiLin/bin/cilin'
#jnius_config.set_classpath('.','/Users/lully/Documents/MSE/JingDong/Code/demo/CiLin/bin/cilin/')


#是否显示调试信息
global in_debug
in_debug = True


CONST_USE_LDA = True


def kmeans(data,k=2):
    def _distance(p1,p2):
        """
            Return Eclud distance between two points.
            p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
            """
        tmp = np.sum((p1-p2)**2)
        return np.sqrt(tmp)
    def _rand_center(data,k):
        """Generate k center within the range of data set."""
        n = data.shape[1] # features
        centroids = np.zeros((k,n)) # init with (0,0)....
        for i in range(n):
            dmin, dmax = np.min(data[:,i]), np.max(data[:,i])
            centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(k)
        return centroids
    
    def _converged(centroids1, centroids2):
        
        # if centroids not changed, we say 'converged'
        set1 = set([tuple(c) for c in centroids1])
        set2 = set([tuple(c) for c in centroids2])
        return (set1 == set2)
    
    
    n = data.shape[0] # number of entries
    centroids = _rand_center(data,k)
    label = np.zeros(n,dtype=np.int) # track the nearest centroid
    assement = np.zeros(n) # for the assement of our model
    converged = False
    
    while not converged:
        old_centroids = np.copy(centroids)
        for i in range(n):
            # determine the nearest centroid and track it with label
            min_dist, min_index = np.inf, -1
            for j in range(k):
                dist = _distance(data[i],centroids[j])
                if dist < min_dist:
                    min_dist, min_index = dist, j
                    label[i] = j
            assement[i] = _distance(data[i],centroids[label[i]])**2
        
        # update centroid
        for m in range(k):
            centroids[m] = np.mean(data[label==m],axis=0)
        converged = _converged(old_centroids,centroids)
    return centroids, label, np.sum(assement)


class Short_Sentence:
    def __init__(self,groupid_init,sentence_init,score_init=None):
        self.groupid = groupid_init
        self.sentence = sentence_init
        self.score = score_init
        self._group_score = 0
        self.user_cut = False
    
    def __str__(self):
        return self.sentence
    
    def __repr__(self):
        return "group id:"+str(self.groupid) + " "+ self.sentence+" score:"+str(self.score)
    
    
    @property
    def group_score(self):
        return self._group_score

    @group_score.setter
    def group_score(self,value):
        self._group_score = value

class Phase:
    def __init__(self,_groupid,_sentence_list):
        self.groupid = _groupid
        self.sentence_list = _sentence_list
        self.avg_score = sum([s.score for s in self.sentence_list])/len(self.sentence_list)

    def phase_string(self):
        result_string = ""
        for ss in self.sentence_list:
            if len(result_string)>0 and result_string[-1] not in [",",".","，","。","！","！","?","？"]:
                result_string += "，"
            result_string += ss.sentence
        return result_string
    
    @property
    def composite_score(self):
        return self._composite_score
    
    @composite_score.setter
    def composite_score(self,value):
        self._composite_score = value
    
    def __repr__(self):
        return "gid:"+str(self.groupid)+",score:"+str(self.avg_score)+ ", composite socre:"+str(self.composite_score) +" phase:"+self.phase_string()

class ContentCreater:

    def __init__(self):
        print("initing content creater........")
        jieba.load_userdict("all_brand.txt")
        stop_word_file ="title_stopwords.txt"
        jieba.analyse.set_stop_words(stop_word_file)
        self.stop_words_set = self.get_stop_words_set(stop_word_file)
        self.checker = SentenceChecker("../../wordvec/model/zhwiki","all_brand.txt")
        self.all_brands = db.read_distinct_brand()
        self.brandList = list(set(self.all_brands))
        self.checker.add_user_words(self.brandList)
        self.model = Word2Vec.load("../../wordvec/new_model/zhwiki")

#        os.environ['CLASSPATH'] = '/Users/lully/Documents/MSE/JingDong/Code/Feature/HanLP-master/hanlp.jar'
#
#        DemoDependencyParser = autoclass('com.hankcs.demo.DemoDependencyParser')
#        self.dependencyParser = DemoDependencyParser()
#
#        WordSim = autoclass('com.hankcs.demo.WordSim')
#        self.wordsim = WordSim()
#        sim = self.wordsim.calcWordsSimilarity("感觉","高端")
#        print("sim:",sim)
#

        self.forbid_words = ['顾客','发票','售后','广告','旗舰店','以上数据','产品参数','仅供参考','理论值','保修','发货','无条件','质保','物流','下单','退货','三包规定','客服','实验室','本店','收货','店铺活动','经销商','免息','分期','订单','活动','好礼','赠送','购机','原装','包装清单','保修卡','说明书','VIP','专享','权益','抢购','购物须知','参数展示','参照官网','测试','赠品','免费获得','专属定制','安装步骤','仅适用']
        self.sentence_filer_words =['不可抗力','为准','领券','购买']

    def get_stop_words_set(self,file_name):
        retList = []
        if util.file_exist(file_name):
            with open(file_name,'r') as file:
                retList = set([line.strip() for line in file])
        return retList

    def load_words_list(self,sentences):
        if in_debug:
            print("共计导入 %d 个停用词"%len(self.stop_words_set))
        word_list = []
        for line in sentences:
    #        s = SnowNLP(line)
    #        if s.sentiments<0.4:
    #            continue

            wordlist = jieba.analyse.textrank(line.strip(), withWeight=True)#,allowPOS=('n', 'vn'))
#            wordlist = jieba.analyse.extract_tags(line.strip(), topK=200, withWeight=True,allowPOS=())#'n', 'vn','a','nz','nh'
#            print("sentence line:"+line+"       words:"+str(wordlist))

            tmp_list =[]
            for w,x in wordlist:
#                print(w)
                if len(w)<1:
                    continue
                tmp_list.append(w)
            if len(tmp_list)<=0:
                continue
            word_list.append([term for term in tmp_list if str(term) not in self.stop_words_set]) #注意这里term是unicode类型，如果不转成str，判断会为假
        return word_list


    def check_sent_one_feature(self,sentence,features):
    #    new_sentence_list = []
    #    for s in ocr_sentence:
    #        new_sentence = ""
    #        lines = re.split(',',s)
    #        for line in lines:
    #            if checker.check_sentence(line):
    #                new_sentence += (line +",")
    #        new_sentence_list.append(new_sentence)

    #    if checker.check_sentence(sentence) == False:
    #        return False

        contains_feature_count = 0
        for f in features:
            if f in sentence:
                contains_feature_count +=1
        if contains_feature_count==1:
            #print("sentence:"+sentence+" features:"+str(features))
            return True

        return False


    def remove_conn(self,sentence):
        if sentence == None or len(sentence)<1:
            return sentence
        conn_list = ["而且","所以","确保","但是","除了","或者","从而","实物","为了","不仅"]
        for cnn in conn_list:
            if cnn in sentence:
                sentence = sentence.replace(cnn,"")

        return sentence

    def choose_sentence(self,sentence_list,sku_brands, brandList,sku_pty_list,all_pyt_list,typename,features):
        sentences_stop_words = ["*","//","购买","我们","]",">","但","理论值","?"]
        #当前sku的属性值要从分类的所有属性值中去除，只要商品特征含有非本商品的其它属性值，则弃之
        if len(sentence_list) <1:
            return ""
#        if len(sentence_list) == 1:
#            return sentence_list[0]

        for i in sku_brands:
            if i in brandList:
                brandList.remove(i)

        for b in brandList:
            if len(b)<=1:
                brandList.remove(b)

        for f in features:
            if f in all_pyt_list:
                all_pyt_list.remove(f)


        for p in sku_pty_list:
            if p in all_pyt_list:
                all_pyt_list.remove(p)
        if typename in all_pyt_list:
            all_pyt_list.remove(typename)
        for st in self.stop_words_set:
            if st in all_pyt_list:
                all_pyt_list.remove(st)

        result_sent = None
        result_sentence_list =[]
        for sentence_obj in sentence_list:
            x = sentence_obj.sentence
            go_next = False

            if in_debug: print("*"*10+x)
            for pty in all_pyt_list:
                if pty in x:
                    go_next = True
                    if in_debug: print("pty:" + pty + "     :"+ x)
                    break
            for brand in brandList:
                if brand in x:
                    go_next = True
                    if in_debug: print("brand:"+brand+ "  :"+x)
                    break
            for sw in sentences_stop_words:
                if sw in x:
                    go_next = True
                    if in_debug:print("stop words:"+sw+": "+x)
                    break
            #TODO 调用语义依存分析， 如果句子中只有施事者，不含有受事者，则说明是半句话，过滤之
            
            if go_next == False:
                result_sent = x
                result_sentence_list.append(x)
                if in_debug: print("found proper sentence:"+x)
    
        if in_debug: print("got sentences count:",len(result_sentence_list))
        score = 0
        if len(result_sentence_list)>0:
            for s in result_sentence_list:
                print("---filter sentence:",s)
                sc = self.score_stence(s)
                if len(s)<len(result_sent) and len(self.remove_punc(s))>6 and len(s)<32:
                    print(sc,self.score_stence(result_sent))
                    if sc > self.score_stence(result_sent):
                        result_sent = s
                        print("---get new proper sentence:",s)

        return self.remove_conn(result_sent)


    def score_stence(self,sentence):
        if len(sentence)<1:
            return 0
        
        n_count = 0
        count = 0
        words = pseg.cut(sentence)
        for word, flag in words:
            if flag == "n":
                n_count +=1
            count +=1
        
        noun_percent = 100*(count-n_count)/count
        return noun_percent
    
    def lda_features(self,sentence_list,numTopics,words_per_topic):
        if len(sentence_list)<=0:
            return []
    #    print(sentence_list)
        word_list = self.load_words_list(sentence_list)
        word_dict = corpora.Dictionary(word_list)  #生成文档的词典，每个词与一个整型索引值对应
    #    word_dict.filter_n_most_frequent(2)#过滤出现频度最高的N个词，一般在处理评论时会有很多“很好”，“不错”等词。

        #dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
        #1.去掉出现次数低于no_below的
        #2.去掉出现次数高于no_above的。注意这个小数指的是百分数
        #3.在1和2的基础上，保留出现频率前keep_n的单词
        #
        #dictionary.filter_tokens(bad_ids=None, good_ids=None)
        #有两种用法，一种是去掉bad_id对应的词，另一种是保留good_id对应的词而去掉其他词。注意这里bad_ids和good_ids都是列表形式
        #
        #dictionary.compacity()
        #在执行完前面的过滤操作以后，可能会造成单词的序号之间有空隙，这时就可以使用该函数来对词典来进行重新排序，去掉这些空隙。
        # word must appear >10 times, and no more than 40% documents
    #    word_dict.filter_extremes(no_below=1, no_above=0.4)

        corpus_list = [word_dict.doc2bow(text) for text in word_list] #词频统计，转化成空间向量格式
        filter(None,corpus_list)
#        print("corpus_list:"+str(corpus_list))
        if corpus_list==None or len(corpus_list)<1:
            return []

#        print("word_list:"+str(word_list))
    #    print("corpus_list:"+str(corpus_list))
        t_start = time.time()
        
        lda = models.ldamodel.LdaModel(corpus=corpus_list,id2word=word_dict,num_topics=numTopics)
        print('-----LDA模型完成，训练时间为\t%.3f秒-----' %(time.time() - t_start))
        
        print("LdaModel.log_perplexity=%.3f" %lda.log_perplexity(corpus_list))
        
        
        document = {}
        review_document = ""
        for i in  lda.show_topics(num_topics=int(numTopics), formatted=False):
            c =0
            for tt in i[1]:
                if c>words_per_topic:
                    continue
                if in_debug: print(tt[0],tt[1])
                if tt[0] in document:
                    continue
                if tt[0].isdigit():
                    continue
                docontinue = False
#                words = pseg.cut(tt[0])
#                for word, flag in words:
#                    if flag == "v":
#                        docontinue = True
#                        print("v"*32+word + "is v")
#                        break
                if docontinue:
                    continue
                document[tt[0]]=tt[1]
                c+=1

        return document


    def get_features(self,raw_sentences,topic_count,words_per_topic):
        return self.lda_features(raw_sentences,topic_count,words_per_topic)

    def get_ocr_feature(self,raw_ocr_sentences):
        return self.get_features(raw_ocr_sentences,2,3)
        
    def get_comment_feature(self,raw_comment_sentences):
        return self.get_features(raw_comment_sentences,3,3)


    def similarity_of_words(self,word1,word2):
    #使用词林来计算相似度（java代码转为python或python 调用java）
        return 0.0



    def remove_punc(self,sentence):
        if len(sentence)<1:
            return sentence
        if sentence[-1] in [",","，",".","。","!","！"]:
            sentence = sentence[:-1]
        if len(sentence)<1:
            return sentence
        if sentence[0] in [",","，",".","。"]:
            sentence = sentence[1:]
        return sentence
    
    
    def cut_words(self,sentences):
        wordlist = []
        for comment in sentences:
            words = jieba.analyse.textrank(comment.strip(), withWeight=True,allowPOS=())
            for w,x in words:
                if len(w)<1:
                    continue
                wordlist.append(w)

        return wordlist
    
    
    def get_raw_sentences(self,skuid,typeid,typename):
        ocr_sentence = db.readOcrs(skuid)
        print("ocr sentences count:"+str(len(ocr_sentence)))
#        if in_debug: [print("=="*3+s) for s in ocr_sentence]

        index = 0
        for phase  in ocr_sentence[:]:
            #BEGIN 去除广告页、客服、特流等图片内容（整张图片去掉）
            forbid_word_count = 0
            for forbid_word in self.forbid_words:
                if forbid_word in phase:
                    forbid_word_count +=1
            if forbid_word_count >=1:
                ocr_sentence.remove(phase)
                print("%"*32,phase)
                continue
            #END
            
            short_sentences = phase.split(",")
            print(short_sentences)
            
            result_phase = ""
            #去除极短的句子，可能是OCR识别有误的文字构成的句子
            for ss in short_sentences[:]:
                last_char = ""
                if len(result_phase)>0:
                    last_char = result_phase[-1]
                if ss == None or len(ss.strip())<4:
                    short_sentences.remove(ss)
                    if last_char in [',','.','。','，']:
                        result_phase = result_phase[:-1]+"。"
                    print("+"*32+ss)
                else:
                    if last_char in [',','.','。','，']:
                        result_phase = result_phase[:-1]+","
                    else:
                        result_phase +=","
                    result_phase +=ss

            ocr_sentence[index] = result_phase
            index +=1
           

        print("ocr_sentence:",ocr_sentence)
        
        comments = db.readComments(skuid)
        print("comment sentences count:"+str(len(comments)))

        return ocr_sentence,comments
    
    
    def combine_ocr_comment_features(self,type_feature,title_features, ocr_features,comment_features):
        print("#"*32)
        print(title_features)
        print(ocr_features)
        print(comment_features)
       
    
        result_features = title_features
        for of in ocr_features.keys():
            if of in result_features:
                continue
            result_features.append(of)

        for cf in comment_features.keys():
            if cf in result_features:
                continue
            result_features.append(cf)

        for tf in type_feature:
            if tf not in result_features:
                result_features.append(tf)
                
        print(result_features)
        print("#"*32)

        return result_features

    
    
    #返回每个feature对应的候选句子列表，以dict格式返回，key为feature，value为list
    def extract_sentences_with_feature(self,features,short_sentences):
#        print("extract_sentences_with_feature",features)
        print("short_sentences:",[ s.sentence for s in short_sentences])

        result_sentences = {}
        
        for feature in features:
            candidate_sents =[]
            matching_list_obj = [s for s in short_sentences if feature in s.sentence]
            #TODO cut attributes not need.
            got_next = False
            for matching_sentence_obj in matching_list_obj:
                if got_next:
                    got_next = False
                    continue
                matching_sentence = matching_sentence_obj.sentence
                if self.checker.check_sentence(matching_sentence)== False:
#                if self.checker.is_correct_sentence(matching_sentence) == False:
                    continue
                if self.check_sent_one_feature(matching_sentence,features):
                    result_sentence_kv = ""
                    if matching_sentence not in candidate_sents:
                        result_sentence_kv = matching_sentence
                    #筛选下一句话
                    next_sentence_index = short_sentences.index(matching_sentence_obj)+1
                    if next_sentence_index < len(short_sentences) and matching_sentence[-1] != "。":
                        next_sent_obj = short_sentences[next_sentence_index]
                        #下一句话只存在于同一个图片中
                        if next_sent_obj.groupid == matching_sentence_obj.groupid:
                            next_sent = next_sent_obj.sentence
                            #下一句中名词占比低于某一值，说明其
                            words = pseg.cut(next_sent)
                            noun_count =0
                            eng_count = 0
                            total_count =0
                            for c,v in words:
                                total_count+=1
                                if v =="n" or v == "vn":
                                    noun_count+=1
                                if v == "eng":# or v=="m":
                                    eng_count +=1
                            noun_percent = noun_count*100/total_count
                            eng_percent = eng_count*100/total_count
                            print("======"+matching_sentence+"============"+next_sent+"====",noun_percent,eng_percent)
                            
                            if noun_percent<=50 and eng_percent<50 and self.checker.check_sentence(next_sent) and self.check_sent_one_feature(next_sent,features)==False and next_sent not in candidate_sents:
                                result_sentence_kv +=next_sent
                                got_next = True
                    result_kv_obj = Short_Sentence(matching_sentence_obj.groupid,result_sentence_kv)
                    
                    candidate_sents.append(result_kv_obj)
            if len(candidate_sents)>0:
                result_sentences[feature] = candidate_sents
                    
        return result_sentences

    def replace_words(self,comment_features,ocr_cut_words):
        for cf,index in enumerate(comment_features):
            most_similar = 0
            ocr_words = cf
            for cutwords in ocr_cut_words:
                sim = self.similarity_of_words(cf,cutwords)
                
                if sim > most_similar:
                    most_similar = sim
                    ocr_words = cutwords
        
            if most_similar>0.6:
                comment_features[index] = ocr_words

        return comment_features
    
    
    #对特征进行聚类运算，合并成cluster_count组
    def cluster_feature(self,features,cluster_ount):
        result_cluster = []
        return features
    
    def sort_cluser_with_priority(self,cluster_features,final_features):
        
        return cluster_features


    #返回：图片编号和含标点的短句子构成的对象列表
    def split_short_sentences(self,raw_sentences):
        raw_ocr_short_sentences =[]

        groupid = 1
        for s in raw_sentences:
            groupid += 1
            #如果是中文前后是空格，则将空格去掉。
            s = util.replace_punc(s," ","")
            s = util.replace_punc(s,"*",",")
            s = util.replace_punc(s,";","。")

            st_list = s.split(",")
            for st in st_list:
                result_phase = ""
                if len(raw_ocr_short_sentences)>0:
                    result_phase = raw_ocr_short_sentences[-1].sentence
                contains_wrong_words = False
                for sfw in self.sentence_filer_words:
                    if sfw in st:
                        contains_wrong_words = True
                        break
                        
                if len(st)<4 or contains_wrong_words:
                    last_char = ""
                    if len(result_phase)>0:#如果有前一句话，则将其句尾以句号结尾
                        last_char = result_phase[-1]
                        if last_char in [',','.','。','，']:
                            result_phase = result_phase[:-1]+"。"
                            raw_ocr_short_sentences[-1].sentence = result_phase
                    
                    continue
                
                #如果中间有断开，说明上下两句是不连贯的，可使用句号连接
                #aaa.aaa!
                st = util.replace_punc(st,".","。")
                st = util.replace_punc(st,"!","。")
#                st = util.replace_punc(st,":","：")
                sst = re.split('[。!]',st)
                for ssst in sst:
                    last_char = ""
                    if len(result_phase)>0:
                        last_char = result_phase[-1]
                    ssst = re.sub("[\(\[].*?[\)\]]", "", ssst)
                    
    
                    if len(ssst)<4:#句子太短了，去之，此句两边的句子不连接，所以用句号连接
                        if last_char in [',','.','。','，']:
                            result_phase = result_phase[:-1]+"。"
                            raw_ocr_short_sentences[-1].sentence= result_phase
                        continue
                    else:
                        if len(sst)>1:
                            result_phase = ssst+"。"
                        else:
                            result_phase = ssst+"，"
#                    raw_ocr_short_sentences.append(self.remove_punc(ssst))
                    short_sentence_obj = Short_Sentence(groupid,result_phase)
#                    print("-"*44,groupid, result_phase)
                    raw_ocr_short_sentences.append(short_sentence_obj)

        return raw_ocr_short_sentences
    
    def sim_sentence(self,sent1,sent2):
        if len(sent1)<=0 or len(sent2)<=0:
            return 0
        
        wordlist1 = jieba.analyse.extract_tags(sent1, withWeight=True)
        wordlist2 = jieba.analyse.extract_tags(sent2, withWeight=True)
#        print(wordlist1)
#        print(wordlist2)

        sims = []
        for w1,x1 in wordlist1:
            max_sim = 0
            for w2,x2 in wordlist2:
                if w1 in self.model.wv.vocab and w2 in self.model.wv.vocab:
                    sim = self.model.similarity(w1,w2)
                    if sim>max_sim:
                        max_sim = sim
            sims.append(max_sim)
    
        for w1,x1 in wordlist2:
            max_sim = 0
            for w2,x2 in wordlist1:
                if w1 in self.model.wv.vocab and w2 in self.model.wv.vocab:
                    sim =self.model.similarity(w1,w2)
                    if sim>max_sim:
                        max_sim = sim
#                else:
#                    print(w1,w2,"not in vocab 2")
            sims.append(max_sim)
        if len(sims)<=0:
            return 0
        
        result_sim = sum(sims)/len(sims)
        return result_sim

    
    #返回候选短句子列表，以dict格式返回，key为groupid，value为list
    #首先调用chekcer打分（句子通顺和错别字等），分值越大越差
    #如果含有lda特征，则加分
    #如果含有数字／英文／特殊符号，则不同程度减分
    #短句打分之后形成短句池
    #计算整个图片的缩合分值,采用min(sentence_score)+avg(sentence_score)综合的方式,计算整个段话中各子句之间的相似度，如果相似度较大，说明不是说的一件事情
    #按照group的分值顺序提取句子，并选择下1句,如果2句以内含有另一个高值短句，则作为下一句一起取得，并标记为已选，在下面选择时不要重复选这句了。
    #最后选择的句子合并成为文章
    def extract_phase(self,features,short_sentences):
#        print("short_sentences:",[ s.sentence for s in short_sentences])

        filtered_sentences = []
        for s in short_sentences[:]:
            s.score = self.checker.eval_sentence(s.sentence)
            print(s.sentence+":::::::::::",s.score)
            if s.score <15:
                filtered_sentences.append(s)
            else:
                if len(filtered_sentences)>0:
#                    filtered_sentences[-1].sentence = filtered_sentences[-1].sentence[:-1]+"。"
                    filtered_sentences[-1].user_cut = True

        if len(filtered_sentences)<=0:
            return None
        #如果句子中含有主题词，鼓励
        for feature in features:
            matching_list_obj = [s for s in filtered_sentences if feature in s.sentence]
            for matching_sentence_obj in matching_list_obj:
                matching_sentence_obj.score  = matching_sentence_obj.score * 0.8
                print(matching_sentence_obj.sentence+":::::::::: s score:",matching_sentence_obj.score)

        #begin construct group list
        group_list = {}
        last_groupid =  filtered_sentences[0].groupid
        
        group_list[last_groupid] =[]
        for idx,s in enumerate( filtered_sentences):
            if last_groupid == s.groupid:
                group_list[last_groupid].append(s)
            else:
                if (idx+1) < len(filtered_sentences):
                    last_groupid = filtered_sentences[idx+1].groupid
                    group_list[last_groupid] = []

        #end construct group list
        phase_list = []
        print("compute group's score:")
        for k,short_sentences_list in group_list.items():
            min_score = 1000000
            avg_score  = sum([ss.score for ss in short_sentences_list])/len(short_sentences_list)
            
            for ss in short_sentences_list:
                if ss.score < min_score:
                    min_score = ss.score
        
            for ss in short_sentences_list:
                ss.group_score = ( min_score/2 + avg_score)
            
            tmp_ph_list = []
            for ss in short_sentences_list:
                tmp_ph_list.append(ss)
                if ss.sentence[-1] in ["！","。","!"] or short_sentences_list[-1]== ss:
                    ph = Phase(ss.groupid,tmp_ph_list)
                    ph.composite_score = self.checker.eval_phase(ph)
                    
                    phase_list.append(ph)
                    tmp_ph_list = []

        for feature in features:
            matching_list_obj = [s for s in  phase_list if feature in s.phase_string()]
            for matching_sentence_obj in matching_list_obj:
                matching_sentence_obj.composite_score  = matching_sentence_obj.composite_score * 0.6
                print(matching_sentence_obj.phase_string()+":::::::::: phase score:",matching_sentence_obj.composite_score)

            
        #按照图片的分值重新排序（由低到高）
        score_sorted_group = sorted(group_list.items(), key=lambda kv: kv[1][0].group_score)
        for key, value in score_sorted_group:
            print ("score_sorted_group : %s:%s, %s" % (key,value, value[0].group_score))

        #将每个图片中的句子以句号分隔，并构建Phase对象（含多个短句），并计算phase的平均分值，最后选择平均分最低的phase构建文章
        sorted_phase_list = sorted(phase_list,key=lambda ph:ph.composite_score)
        for sph in sorted_phase_list:
            print("sorted_phase_list:%s"%(sph))



        print("$$$$$$"*3)
#        result_sentences_list = []
#        for feature in features[:]:
#            for s in sorted_phase_list:
#                phase_string =  s.phase_string()
#                if feature in phase_string and phase_string not in result_sentences_list:
#                    result_sentences_list.append(phase_string)
#                    #同一句子中含有多个关键词
#                    for f in features:
#                        if f in phase_string:
#                            features.remove(f)
#                    break
#
#        print(result_sentences_list)
#        #重新计算句子的lda主题
#        words = self.lda_features(result_sentences_list,2,3)
#        result_words = []
#        for w in words:
#            if w not in result_words:
#                result_words.append(w)
#
#        new_result = []
#        for feature in result_words[:]:
#            for s in result_sentences_list:
#                phase_string =  s
#                if feature in phase_string and phase_string not in new_result:
#                    new_result.append(phase_string)
#                    #同一句子中含有多个关键词
#                    for f in result_words:
#                        if f in phase_string:
#                            result_words.remove(f)
#                            break
#
#        print("$$$$$$$$$$$$$$$$$$$$$")
#        print(new_result)



        #begin phase分词后计算各分句的相似度，移除相似度超过0.5的高分值句子
        new_phase_list = []
        need_remove_phases = {}
        for idx,rs in enumerate(sorted_phase_list[:]):
            max_sim = 0
            max_sim_sentence = ""
            for idx2,rs2 in enumerate(sorted_phase_list[:]):
                if rs == rs2 or idx2 <= idx:
                    continue
                
                sim = self.sim_sentence(rs.phase_string(),rs2.phase_string())

                if sim > 0.4:
                    print(max_sim,rs,"|||",rs2)
                    if idx not in need_remove_phases:
                        need_remove_phases[idx] =[]
                    if rs not in need_remove_phases[idx]:
                        need_remove_phases[idx].append(rs)
                    if rs2 not in need_remove_phases[idx]:
                        need_remove_phases[idx].append(rs2)
                else:
                    new_phase_list.append(rs)

        real_remove_phases = []
        for k,need_remove_list in need_remove_phases.items():
            min_score = need_remove_list[0].composite_score
            remain_ph = None
            for ph in need_remove_list[:]:
                if ph.composite_score > min_score:
                    if ph not in real_remove_phases:
                        real_remove_phases.append(ph)
                else:
                    min_score = ph.composite_score
        #end

        print("去除相似phase后:")
        sorted_phase_list = [sp for sp in sorted_phase_list if sp not in real_remove_phases]
        print(sorted_phase_list)

        return [s.phase_string() for s in sorted_phase_list]
#        return result_sentences_list


    
    #生成摘要内容
    def gen_content(self,skuid,typeid,features,typename,brand_name):
        print("/"*64,datetime.datetime.now())
        combined_content = ""
        complete = False
        print("title features:"+str(features))
        if typename not in self.stop_words_set:
            self.stop_words_set.add(typename)
        
        if brand_name not in self.stop_words_set:
            self.stop_words_set.add(brand_name)
                                 
        title_features = features
        
        raw_ocr_sentences,raw_comment_sentences = self.get_raw_sentences(skuid,typeid,typename)
        #ocr句子分成短句
        raw_ocr_short_sentences = self.split_short_sentences(raw_ocr_sentences)
        if raw_ocr_short_sentences == None or len(raw_ocr_short_sentences)<=0:
            return ""
        
#        #begin write all short sentences to file
#        short_list = []
#        short_list.append("sku:"+str(skuid))
#        last_groupid = raw_ocr_short_sentences[0].groupid
#        last_sentence = ""
#        for short_obj in raw_ocr_short_sentences:
#            if last_groupid == short_obj.groupid:
#                last_sentence += short_obj.sentence
#            else:
#                if len(last_sentence)>0:
#                    short_list.append(last_sentence)
#                last_sentence = str(short_obj.groupid)+":"
#                last_groupid = short_obj.groupid
#
#        if len(short_list)>0:
#            util.writeList2File("shorts.txt",short_list,'a')
#        #end write all short sentences to file

        raw_short_sentences_str = "|".join((str(ss) for ss in raw_ocr_short_sentences))
        
        print("raw_ocr_short_sentences:",raw_short_sentences_str)
        if len(raw_short_sentences_str)<500:#总字数少于500字，不适全使用机器的方式
            return ""
        
        ocr_features = self.get_ocr_feature(raw_ocr_sentences)
        print(ocr_features)
        if len(ocr_features)<1:
            return ""
        comment_features = self.get_comment_feature(raw_comment_sentences)
        
        ocr_cut_words = self.cut_words(raw_ocr_sentences)
        
        #计算每个comment特征与分词的相似度，使用ocr的分词作为特征词
        comment_features = self.replace_words(comment_features,ocr_cut_words)
        
        type_features = ['机身','功能','设计','芯片','电池','前置','设计','摄像头','像素','屏幕']
        
        final_features = self.combine_ocr_comment_features(type_features,title_features, ocr_features,comment_features)
        
        phase_list = self.extract_phase(final_features,raw_ocr_short_sentences)
        
        #begin test
        article = ""
        for ss in phase_list:
            article += ss
            if len(article)>80:
                break
        
        util.writeList2File("new_article.txt",[str(skuid) + ": " + ''.join(phase_list)],'a')
        #end test

        return article
        
        #取ocr中句子
        feature_sentences_dict = self.extract_sentences_with_feature(final_features,raw_ocr_short_sentences)
        
        print("feature_sentences_dict:")
        for k,v in feature_sentences_dict.items():
            print(k,v)
        
        #特征词进行相似性聚类
        cluster_features = self.cluster_feature(feature_sentences_dict.keys(),5)
        #按照主题概率进行组内排序
        cluster_features = self.sort_cluser_with_priority(cluster_features,final_features)
        
        sku_pty_list = db.read_sku_pty(skuid)
        if typeid == "9435":
            choosed_pty_list = ['香型','品牌', '规格','包装','省份','产地','酿造工艺','等级','净含量']
        if typeid == "655":
            choosed_pty_list = ['CPU型号','CPU品牌','型号','品牌','热点','常用功能','后置摄像头','操作系统','屏幕材质类型']
         
        all_pyt_list = db.read_pry_typeid(skuid,typeid,choosed_pty_list)
        sku_brands = db.read_distinct_brand(skuid)
        sku_brands = list(set(sku_brands))
        all_brands = db.read_distinct_brand(typeid)
        brandList = list(set(all_brands))
         
        result_article = ""
        for cf in cluster_features:
            sentences_obj = feature_sentences_dict[cf]
            choosed_sentence = self.choose_sentence(sentences_obj,sku_brands,brandList,sku_pty_list,all_pyt_list,typename,cluster_features)
            if choosed_sentence != None:
                new_sentence =util.replace_punc(choosed_sentence,".",",")
                new_sentence =util.replace_punc(new_sentence,"。",",")
                
                result_article += new_sentence
            if len(result_article)>80:
                break
                 
        result_article = result_article[:-1]+"."
        return result_article


