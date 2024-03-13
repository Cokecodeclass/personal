import re
from types import MethodType, FunctionType
import pandas as pd
# import pkuseg
# import jieba
import thulac

import synonyms
import random
from random import shuffle
from chinese2digits.chinese2digits import takeChineseNumberFromString
import time
import os
import sys
import itertools
root = os.path.dirname(os.path.abspath(__file__))
from fake_driver.config import stop_words_path, didi_dict_path, AIR_NO_REG1, AIR_NO_REG2, WX_PHONE_REG, TRAIN_NO, PLATE_NO_REG1, PLATE_NO_STD, CHANNEL_NO1
# from fake_driver.utils.wrapt_timeout_decorator import timeout

def set_seg_obj(dict_path=didi_dict_path):
    # seg_obj = pkuseg.pkuseg(user_dict=dict_path)
    seg_obj = thulac.thulac(user_dict=dict_path, seg_only=True)

    return seg_obj


# import pkuseg
# seg_obj = pkuseg.pkuseg(user_dict=didi_dict_path)
seg_obj = thulac.thulac(user_dict=didi_dict_path, seg_only=True)
import re
from types import MethodType, FunctionType


def clean_txt(raw):
    # keep available char
    raw = raw.upper()
#     try:
    fil = re.compile('([\[\(（].*[\]\)）])')
    raw = fil.sub(' ', raw)
    fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    raw = fil.sub(' ', raw)
    fil = re.compile(WX_PHONE_REG)
    raw = fil.sub(' WXPHONE ', raw)
    fil = re.compile(CHANNEL_NO1)
    raw = fil.sub(' CHANNEL ', raw)
    fil = re.compile(PLATE_NO_STD)
    raw = fil.sub(' PLATENO ', raw)
    fil = re.compile('([车牌尾号是]{2,5}([A-Z]{1}[A-Z0-9]{3,6})|([A-Z]{1}[A-Z0-9]{3,6})[车牌尾号]{2,4})|车[牌尾号是]{1,4}[A-Z0-9]{3,6}')
    raw = re.sub(fil, '\g<1> PLATENO ', raw)
    fil = re.compile(AIR_NO_REG2)
    raw = fil.sub(' AIRNO ', raw)
    fil = re.compile(r'(\d+(\s\d)+)')
    raw = fil.sub(' DIGIT ', raw)
    return raw

def seg(sentence, sw, apply=None):
    if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
        sentence = apply(sentence)
    seg_v1 = seg_obj.fast_cut(sentence, text=True)
    seg_res = seg_v1.split()
    seg_res = ' '.join([i for i in seg_res if i.strip() and i not in sw])
    return seg_res

def stop_words(stop_words_path):
    with open(stop_words_path, 'r', encoding='utf-8') as swf:
        return [line.strip() for line in swf]
# @timeout(1, use_signals=False)
def pre_seg_contents_for_lime(content):
    res = seg(fil_r.sub(' ', content.upper()), stop_words(stop_words_path), apply=clean_txt)
    return res


########################################################################
# 同义词替换
# 替换一个语句中的n个单词为其同义词
########################################################################
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stws]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    return synonyms.nearby(word)[0]


########################################################################
# 随机插入
# 随机在语句中插入n个词
########################################################################
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

########################################################################
# 随机删除
# 以概率p删除语句中的词
########################################################################
def random_deletion(words, p):

    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words


########################################################################
#EDA函数
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    seg_list = seg(fil_r.sub(' ', sentence.lower()), stop_words(stop_words_path), apply=clean_txt)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    #print(words, "\n")


    #同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

    #随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

    #随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    #随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))

    #print(augmented_sentences)
    shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(seg_list)

    return augmented_sentences

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        #全角空格直接转换
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

class FakerPreprocessor(object):
    def __init__(self, stw_path=stop_words_path, dict_path=didi_dict_path):
        self.stw_path = stw_path
        self.dict_path = dict_path
        self.seg_obj = set_seg_obj(dict_path)
        self.stop_words = stop_words(stw_path)

    def seg(self, sentence, sw, apply=None):
        if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
            sentence = apply(sentence)
        seg_v1 = self.seg_obj.fast_cut(sentence, text=True)
        seg_res = seg_v1.split()
        seg_res = ' '.join([i for i in seg_res if i.strip() and i not in sw])
        return seg_res

    def process_content_char(self, raw):
        # fil_r = re.compile(r'ｌ')
        fil = re.compile('([\[\(（].*[\]\)）])')
        raw = fil.sub(' ', raw)
        fil_r = re.compile(r'@1|@2|\$\-1|\|\||\$1')
        raw = fil_r.sub(' ', raw)
        raw = strQ2B(raw)
        # print(raw)
        fil = re.compile(r'\d{1,2}:\d{2}')
        raw = fil.sub("约定时间", raw)
        # print(raw)
        fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5@\|\$]+")
        raw = fil.sub(' ', raw)
        # print(raw)
        raw = raw.upper()


        fil = re.compile(r"OK[u4e00-\u9fa5]{1,3}[分钟点多]{1,4}")
        raw = fil.sub("约定时间", raw)
        # 转换缩写
        fil = re.compile(r'(KFC|KTV|BRT|SUV|HELLO|HI|SONY|ETC|MALL|APP|TCL|OK|SORRY)')
        raw = fil.sub('英文缩写', raw)

        # 防止错误转换
        raw = raw.replace('1万', '1=万')
        raw = raw.replace('1千', '1=千')
        raw = raw.replace('1百', '1=百')
        raw = raw.replace('1十', '1=十')
        s_reg = re.compile(r'(?=.*[A-Za-z0-9])(?=.*\d)[A-Za-z\d]{2,50}')
        t_find = re.findall(s_reg, raw)
        ts = re.split(s_reg, raw)
        if t_find:
            # print(ts, t_find)
            n_raw = list()
            for tr in ts:
                nr = takeChineseNumberFromString(tr, percentConvert=False, skipError=True)
                n_raw.append(nr['replacedText'])
            ns = list(itertools.chain.from_iterable(zip(n_raw, t_find)))
            ns.append(ts[-1])
            # print(ns)
            res = takeChineseNumberFromString(raw, percentConvert=False, skipError=True)
            raw = res['replacedText']
            raw = "".join(ns)
            # print(raw)
        else:
            # print('not find')
            res = takeChineseNumberFromString(raw, percentConvert=False, skipError=True)
            raw = res['replacedText']
        raw = raw.replace('SUV', '运动型车')
        # BUG 加内容类型。
        # raw = raw.replace('标志', '标致')
        raw = raw.replace('MINI', '宝马')
        raw = raw.replace('JEEP', '吉普')
        raw = raw.replace('YAMAHA', '雅马哈')
        raw = raw.replace('1辆', '一辆')
        raw = raw.replace('1样', '一样')
        raw = raw.replace('1个', '一个')
        raw = raw.replace('1会', '一会')
        raw = raw.replace('1下', '一下')
        raw = raw.replace('1汽', '一汽')
        raw = raw.replace('50铃', '五十铃')
        raw = raw.replace('5菱', '五菱')
        raw = raw.replace('7座', '七座')
        raw = raw.replace('3轮', '三轮')
        raw = raw.replace('白车', '白色')
        raw = raw.replace('黑车', '黑色')
        raw = raw.replace('红车', '红色')
        raw = raw.replace('蓝车', '蓝色')
        raw = raw.replace('黄车', '黄色')
        raw = raw.replace('雪弗兰', '雪佛兰')
        raw = raw.replace('雪佛莱', '雪佛兰')
        # print(raw)
        return raw

    # @timeout(1, use_signals=False)
    def pre_seg_contents_for_lime(self, content):
        res = seg(fil_r.sub(' ', content.upper()), self.stop_words, apply=clean_txt)
        return res

    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        seg_list = self.seg(fil_r.sub(' ', sentence.upper()), self.stop_words, apply=clean_txt)
        seg_list = " ".join(seg_list)
        words = list(seg_list.split())
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(num_aug/4)+1
        n_sr = max(1, int(alpha_sr * num_words))
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        #print(words, "\n")


        #同义词替换sr
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

        #随机插入ri
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

        #随机交换rs
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

        #随机删除rd
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

        #print(augmented_sentences)
        shuffle(augmented_sentences)

        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        augmented_sentences.append(seg_list)

        return augmented_sentences


fil_r = re.compile(r'@1|@2|\$\-1|\|\||\$1|ｌ')
# res = seg(fil_r.sub(' ', content.upper()), stop_words(), apply=clean_txt)
# print(res)


def _format_label(a):
    return '__label__'+str(a)


def seg_and_format_train_data(path):
    df = pd.read_csv(path, header=None, sep='\t')
    df.columns = ['contents', 'label']
    df['label'] = df['label'].apply(lambda x: _format_label(x))
    fil_r = re.compile(r'@1|@2|\$\-1|\|\||\$1')
    df['contents'] = df['contents'].apply(lambda x: seg(
        fil_r.sub(' ', x.upper()), stop_words(stop_words_path), apply=clean_txt))
    return df


def transform2input_data(input_path):
    res_df = seg_and_format_train_data(input_path)
    save_path = input_path+'.input'
    new_res_df = res_df[['label', 'contents']]
    new_res_df.to_csv(save_path, index=None, header=None, sep='\t')
    print('done. file is saving in {}'.format(save_path))
    return save_path
