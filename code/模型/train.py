#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_model.py
@Time    :   2020/11/04 17:01:19
@Author  :   Charles Lai
@Version :   1.0
@Contact :   charleslaihongchang@didiglobal.com
@Desc    :   None
'''

#拆分司机角色建模

# here put the import lib
import numpy as np
import fasttext
import re
from datetime import  datetime
import time

from sklearn.metrics import roc_curve, auc, roc_auc_score

from fake_driver.feature.car_nlp_feature import CarNlpFeature

from fake_driver.preprocess.preprocess import FakerPreprocessor, clean_txt, seg_obj, takeChineseNumberFromString

from fake_driver.feature import CarMessageTemplate

from collections import OrderedDict
import pandas as pd
import joblib
import gc
from tqdm import *
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ProcessPool
from multiprocessing import TimeoutError
from sklearn.metrics import classification_report
import xgboost as xgb
from warmup.metrics import BinaryClassEstimate



class NewFeature(CarNlpFeature):
    def __init__(self, ext_api = None, logger_handler=None):
        CarNlpFeatureMulti.__init__(self, ext_api=ext_api, logger_handler=logger_handler)
    def init(self):
        self.tmp_desc = {}
        self.template = CarMessageTemplate()
        self.preprocessor = FakerPreprocessor()
        ###新特征的计算可能依赖之前的某些特征值，故需要保证特征组之间计算的顺序
        self.feature_map = OrderedDict({
            101000:self.get_content_statics,
            102000:self.get_im_content_feature_multithread,
            104000:self.get_psger_im_content_feature_multithread,
            105000:self.get_driver_im_content_feature_multithread,
            'external': self.get_external_feature,
        })


def pre_seg_contents_for_lime(content):
    res = fake_p.pre_seg_contents_for_lime(content)
    return res

def _format_label(a):
    """
    0 相符
    1 车不符

    """
    if isinstance (a, int):
        return '__label__'+str(a)
    if a.startswith('__label__'):
        return a
    else:
        return '__label__'+str(a)

def _re_label_int(a):
    return int(a[-1])

def seg_and_format_train_data(path):
    df = pd.read_csv(path, header=None, sep='\t')
    df.columns = ['contents', 'label']
    df['label'] = df['label'].apply(lambda x: _format_label(x))
    df['contents'] = df['contents'].apply(lambda x: fake_p.pre_seg_contents_for_lime(fil_r.sub(' ', x.upper())))
    return df

def transform2input_data(input_path):
    res_df = seg_and_format_train_data(input_path)
    save_path = input_path+'.input'
    new_res_df = res_df[['label', 'contents']]
    new_res_df.to_csv(save_path, index=None, header=None, sep='\t')
    print('done. file is saving in {}'.format(save_path))
    return save_path

def split_contents(contents):
    content_list = contents.split('||')
    driver_contents = [c for c in content_list if '@2' in c]
    d_contents = "||".join(driver_contents)
    return d_contents


def preprocess_data_set(path, output_path, frac=1):
    df = pd.read_csv(path, sep='\t')
    df = df[df.label!=-1]
    if 0 < frac < 1:
        df = df.sample(frac=frac, random_state=1024)
    # df.columns = ['d_pid', 'contents', 'cur_date', 'final_gvid', 'plate_no', 'color', 'brand_desc', 'uuid', 'label', 'cate']
    df.columns = ['label',
 'o_id',
 'd_id',
 'contents',
 'plate_no',
 'color',
 'brand_desc',
 'cur_date',
 'randkey',
 'order_id',
 'driver_id',
 'passenger_id',
 'route_id',
 'max_depart_time',
 'call_dt',
 'depart_hour',
 'depart_weekday',
 'est_dis',
 'busi_type',
 'is_cross_city_flag',
 'is_station_flag',
 'extra_info',
 'is_extra_info',
 'is_luggage',
 'passenger_num',
 'start_lng',
 'start_lat',
 'invite_time',
 'assigned_time',
 'answer_lat',
 'answer_lng',
 'prepared_lng',
 'prepared_lat',
 'd_gender',
 'p_gender',
 'man_driver_woman_pas',
 'seat_cnt',
 'friends_cnt',
 'depart_invite_dt',
 'answer_start_distance',
 'answer_start_speed',
 'prepared_start_distance',
 'd_rnid',
 'arrive_order_cnt_90d',
 'arrive_order_cnt_60d',
 'arrive_order_cnt_30d',
 'arrive_order_cnt_7d',
 'arrive_station_order_cnt_90d',
 'arrive_station_order_cnt_60d',
 'arrive_station_order_cnt_30d',
 'arrive_station_order_cnt_7d',
 'arrive_cross_order_cnt_90d',
 'arrive_cross_order_cnt_60d',
 'arrive_cross_order_cnt_30d',
 'arrive_cross_order_cnt_7d',
 'arrive_carp_succ_order_cnt_90d',
 'arrive_carp_succ_order_cnt_60d',
 'arrive_carp_succ_order_cnt_30d',
 'arrive_carp_succ_order_cnt_7d',
 'arrive_free_order_cnt_90d',
 'arrive_free_order_cnt_60d',
 'arrive_free_order_cnt_30d',
 'arrive_free_order_cnt_7d',
 'arrive_report_order_cnt_90d',
 'arrive_report_order_cnt_60d',
 'arrive_report_order_cnt_30d',
 'arrive_report_order_cnt_7d',
 'arrive_dis50_order_cnt_90d',
 'arrive_dis50_order_cnt_60d',
 'arrive_dis50_order_cnt_30d',
 'arrive_dis50_order_cnt_7d',
 'arrive_dis100_order_cnt_90d',
 'arrive_dis100_order_cnt_60d',
 'arrive_dis100_order_cnt_30d',
 'arrive_dis100_order_cnt_7d',
 'arrive_night_order_cnt_90d',
 'arrive_night_order_cnt_60d',
 'arrive_night_order_cnt_30d',
 'arrive_night_order_cnt_7d',
 'arrive_dis50_night_order_cnt_90d',
 'arrive_dis50_night_order_cnt_60d',
 'arrive_dis50_night_order_cnt_30d',
 'arrive_dis50_night_order_cnt_7d',
 'arrive_start_distance_1km_90d',
 'arrive_start_distance_1km_60d',
 'arrive_start_distance_1km_30d',
 'arrive_start_distance_1km_7d',
 'finish_dest_distance_1km_90d',
 'finish_dest_distance_1km_60d',
 'finish_dest_distance_1km_30d',
 'finish_dest_distance_1km_7d',
 'arrive_good_tag_order_cnt_90d',
 'arrive_good_tag_order_cnt_60d',
 'arrive_good_tag_order_cnt_30d',
 'arrive_good_tag_order_cnt_7d',
 'arrive_bad_tag_order_cnt_90d',
 'arrive_bad_tag_order_cnt_60d',
 'arrive_bad_tag_order_cnt_30d',
 'arrive_bad_tag_order_cnt_7d',
 'arrive_undefine_pas_order_cnt_90d',
 'arrive_undefine_pas_order_cnt_60d',
 'arrive_undefine_pas_order_cnt_30d',
 'arrive_undefine_pas_order_cnt_7d',
 'arrive_person_car_complaint_cut_90d',
 'arrive_person_car_complaint_cut_60d',
 'arrive_person_car_complaint_cut_30d',
 'arrive_person_car_complaint_cut_7d',
 'arrive_person_car_complaint_cut_real_90d',
 'arrive_person_car_complaint_cut_real_60d',
 'arrive_person_car_complaint_cut_real_30d',
 'arrive_person_car_complaint_cut_real_7d',
 'arrive_person_complaint_cut_90d',
 'arrive_person_complaint_cut_60d',
 'arrive_person_complaint_cut_30d',
 'arrive_person_complaint_cut_7d',
 'arrive_person_complaint_cut_real_90d',
 'arrive_person_complaint_cut_real_60d',
 'arrive_person_complaint_cut_real_30d',
 'arrive_person_complaint_cut_real_7d',
 'arrive_car_complaint_cut_90d',
 'arrive_car_complaint_cut_60d',
 'arrive_car_complaint_cut_30d',
 'arrive_car_complaint_cut_7d',
 'arrive_car_complaint_cut_real_90d',
 'arrive_car_complaint_cut_real_60d',
 'arrive_car_complaint_cut_real_30d',
 'arrive_car_complaint_cut_real_7d',
 'arrive_person_complaint_90d',
 'arrive_person_complaint_60d',
 'arrive_person_complaint_30d',
 'arrive_person_complaint_7d',
 'arrive_person_complaint_real_90d',
 'arrive_person_complaint_real_60d',
 'arrive_person_complaint_real_30d',
 'arrive_person_complaint_real_7d',
 'arrive_person_cut_90d',
 'arrive_person_cut_60d',
 'arrive_person_cut_30d',
 'arrive_person_cut_7d',
 'arrive_person_cut_real_90d',
 'arrive_person_cut_real_60d',
 'arrive_person_cut_real_30d',
 'arrive_person_cut_real_7d',
 'arrive_car_complaint_90d',
 'arrive_car_complaint_60d',
 'arrive_car_complaint_30d',
 'arrive_car_complaint_7d',
 'arrive_car_complaint_real_90d',
 'arrive_car_complaint_real_60d',
 'arrive_car_complaint_real_30d',
 'arrive_car_complaint_real_7d',
 'arrive_car_cut_90d',
 'arrive_car_cut_60d',
 'arrive_car_cut_30d',
 'arrive_car_cut_7d',
 'arrive_car_cut_real_90d',
 'arrive_car_cut_real_60d',
 'arrive_car_cut_real_30d',
 'arrive_car_cut_real_7d',
 'arrive_cond_plateno_not_same_90d',
 'arrive_cond_plateno_not_same_60d',
 'arrive_cond_plateno_not_same_30d',
 'arrive_cond_plateno_not_same_7d',
 'arrive_cond_brand_not_same_90d',
 'arrive_cond_brand_not_same_60d',
 'arrive_cond_brand_not_same_30d',
 'arrive_cond_brand_not_same_7d',
 'arrive_cond_series_not_same_90d',
 'arrive_cond_series_not_same_60d',
 'arrive_cond_series_not_same_30d',
 'arrive_cond_series_not_same_7d',
 'arrive_cond_change_car_90d',
 'arrive_cond_change_car_60d',
 'arrive_cond_change_car_30d',
 'arrive_cond_change_car_7d',
 'arrive_cond_color_not_same_90d',
 'arrive_cond_color_not_same_60d',
 'arrive_cond_color_not_same_30d',
 'arrive_cond_color_not_same_7d',
 'arrive_re_rule_order_cnt_90d',
 'arrive_re_rule_order_cnt_60d',
 'arrive_re_rule_order_cnt_30d',
 'arrive_re_rule_order_cnt_7d',
 'arrive_station_order_rate_90d',
 'arrive_station_order_rate_60d',
 'arrive_station_order_rate_30d',
 'arrive_station_order_rate_7d',
 'arrive_cross_order_rate_90d',
 'arrive_cross_order_rate_60d',
 'arrive_cross_order_rate_30d',
 'arrive_cross_order_rate_7d',
 'arrive_carp_succ_order_rate_90d',
 'arrive_carp_succ_order_rate_60d',
 'arrive_carp_succ_order_rate_30d',
 'arrive_carp_succ_order_rate_7d',
 'arrive_free_order_rate_90d',
 'arrive_free_order_rate_60d',
 'arrive_free_order_rate_30d',
 'arrive_free_order_rate_7d',
 'arrive_report_order_rate_90d',
 'arrive_report_order_rate_60d',
 'arrive_report_order_rate_30d',
 'arrive_report_order_rate_7d',
 'arrive_dis50_order_rate_90d',
 'arrive_dis50_order_rate_60d',
 'arrive_dis50_order_rate_30d',
 'arrive_dis50_order_rate_7d',
 'arrive_dis100_order_rate_90d',
 'arrive_dis100_order_rate_60d',
 'arrive_dis100_order_rate_30d',
 'arrive_dis100_order_rate_7d',
 'arrive_night_order_rate_90d',
 'arrive_night_order_rate_60d',
 'arrive_night_order_rate_30d',
 'arrive_night_order_rate_7d',
 'arrive_dis50_night_order_rate_90d',
 'arrive_dis50_night_order_rate_60d',
 'arrive_dis50_night_order_rate_30d',
 'arrive_dis50_night_order_rate_7d',
 'arrive_start_distance_1km_rate_90d',
 'arrive_start_distance_1km_rate_60d',
 'arrive_start_distance_1km_rate_30d',
 'arrive_start_distance_1km_rate_7d',
 'finish_dest_distance_1km_rate_90d',
 'finish_dest_distance_1km_rate_60d',
 'finish_dest_distance_1km_rate_30d',
 'finish_dest_distance_1km_rate_7d',
 'arrive_good_tag_order_rate_90d',
 'arrive_good_tag_order_rate_60d',
 'arrive_good_tag_order_rate_30d',
 'arrive_good_tag_order_rate_7d',
 'arrive_bad_tag_order_rate_90d',
 'arrive_bad_tag_order_rate_60d',
 'arrive_bad_tag_order_rate_30d',
 'arrive_bad_tag_order_rate_7d',
 'arrive_undefine_pas_order_rate_90d',
 'arrive_undefine_pas_order_rate_60d',
 'arrive_undefine_pas_order_rate_30d',
 'arrive_undefine_pas_order_rate_7d',
 'arrive_person_car_complaint_cut_rate_90d',
 'arrive_person_car_complaint_cut_rate_60d',
 'arrive_person_car_complaint_cut_rate_30d',
 'arrive_person_car_complaint_cut_rate_7d',
 'arrive_person_car_complaint_cut_real_rate_90d',
 'arrive_person_car_complaint_cut_real_rate_60d',
 'arrive_person_car_complaint_cut_real_rate_30d',
 'arrive_person_car_complaint_cut_real_rate_7d',
 'arrive_person_complaint_cut_rate_90d',
 'arrive_person_complaint_cut_rate_60d',
 'arrive_person_complaint_cut_rate_30d',
 'arrive_person_complaint_cut_rate_7d',
 'arrive_person_complaint_cut_real_rate_90d',
 'arrive_person_complaint_cut_real_rate_60d',
 'arrive_person_complaint_cut_real_rate_30d',
 'arrive_person_complaint_cut_real_rate_7d',
 'arrive_car_complaint_cut_rate_90d',
 'arrive_car_complaint_cut_rate_60d',
 'arrive_car_complaint_cut_rate_30d',
 'arrive_car_complaint_cut_rate_7d',
 'arrive_car_complaint_cut_real_rate_90d',
 'arrive_car_complaint_cut_real_rate_60d',
 'arrive_car_complaint_cut_real_rate_30d',
 'arrive_car_complaint_cut_real_rate_7d',
 'arrive_person_complaint_rate_90d',
 'arrive_person_complaint_rate_60d',
 'arrive_person_complaint_rate_30d',
 'arrive_person_complaint_rate_7d',
 'arrive_person_complaint_real_rate_90d',
 'arrive_person_complaint_real_rate_60d',
 'arrive_person_complaint_real_rate_30d',
 'arrive_person_complaint_real_rate_7d',
 'arrive_person_cut_rate_90d',
 'arrive_person_cut_rate_60d',
 'arrive_person_cut_rate_30d',
 'arrive_person_cut_rate_7d',
 'arrive_person_cut_real_rate_90d',
 'arrive_person_cut_real_rate_60d',
 'arrive_person_cut_real_rate_30d',
 'arrive_person_cut_real_rate_7d',
 'arrive_car_complaint_rate_90d',
 'arrive_car_complaint_rate_60d',
 'arrive_car_complaint_rate_30d',
 'arrive_car_complaint_rate_7d',
 'arrive_car_complaint_real_rate_90d',
 'arrive_car_complaint_real_rate_60d',
 'arrive_car_complaint_real_rate_30d',
 'arrive_car_complaint_real_rate_7d',
 'arrive_car_cut_rate_90d',
 'arrive_car_cut_rate_60d',
 'arrive_car_cut_rate_30d',
 'arrive_car_cut_rate_7d',
 'arrive_car_cut_real_rate_90d',
 'arrive_car_cut_real_rate_60d',
 'arrive_car_cut_real_rate_30d',
 'arrive_car_cut_real_rate_7d',
 'arrive_cond_plateno_not_same_rate_90d',
 'arrive_cond_plateno_not_same_rate_60d',
 'arrive_cond_plateno_not_same_rate_30d',
 'arrive_cond_plateno_not_same_rate_7d',
 'arrive_cond_brand_not_same_rate_90d',
 'arrive_cond_brand_not_same_rate_60d',
 'arrive_cond_brand_not_same_rate_30d',
 'arrive_cond_brand_not_same_rate_7d',
 'arrive_cond_series_not_same_rate_90d',
 'arrive_cond_series_not_same_rate_60d',
 'arrive_cond_series_not_same_rate_30d',
 'arrive_cond_series_not_same_rate_7d',
 'arrive_cond_change_car_rate_90d',
 'arrive_cond_change_car_rate_60d',
 'arrive_cond_change_car_rate_30d',
 'arrive_cond_change_car_rate_7d',
 'arrive_cond_color_not_same_rate_90d',
 'arrive_cond_color_not_same_rate_60d',
 'arrive_cond_color_not_same_rate_30d',
 'arrive_cond_color_not_same_rate_7d',
 'arrive_re_rule_order_rate_90d',
 'arrive_re_rule_order_rate_60d',
 'arrive_re_rule_order_rate_30d',
 'arrive_re_rule_order_rate_7d',
 'pt',
 'd_rnid_1',
 'reg_dt_max',
 'reg_dt_min',
 'employ',
 'age_max',
 'age_min',
 'dishonest_status',
 'black_flag',
 '3m_route_time_type_max',
 'week_dri_flag',
 'is_profession',
 '15d_offline_flag',
 '15d_cancel_cash_flag',
 '30_high_cancel_sta_flag',
 'resident_reg_sample_city_max',
 'resident_reg_sample_city_min',
 'is_gulf_driver',
 'has_emergency',
 'emergency_is_sfc',
 'emergency_is_wyc',
 'car_cnt_sum',
 'car_age_max',
 'car_age_min',
 'car_price_max',
 'car_price_min',
 'car_city_contains_resident_reg',
 'car_city_not_contains_resident_reg',
 'travel_license_contains_card',
 'is_valid_status_sum',
 'travel_auth_state_min',
 'person_car_auth_state_min',
 'blacked_cnt',
 'blacked_woman_cnt',
 'pt_1',
 'd_rnid_2',
 'answer_order_cnt',
 'cancel_order_cnt',
 'driver_cancel_order_cnt',
 'pas_cancel_order_cnt',
 'cancel_rate',
 'driver_cancel_rate',
 'pas_cancel_rate',
 'dt_2']
    df['contents']= df['contents'].apply(lambda x:split_contents(x))
    # raw_df = df[['final_gvid', 'd_pid', 'contents', 'plate_no', 'color','brand_desc', 'cur_date', 'label']]
    raw_df = df[['label',
 'o_id',
 'd_id',
 'contents',
 'plate_no',
 'color',
 'brand_desc',
 'cur_date',
 'depart_hour',
 'depart_weekday',
 'est_dis',
 'busi_type',
 'is_cross_city_flag',
 'is_station_flag',
 'is_extra_info',
 'is_luggage',
 'passenger_num',
 'd_gender',
 'p_gender',
 'man_driver_woman_pas',
 'seat_cnt',
 'friends_cnt',
 'depart_invite_dt',
 'answer_start_distance',
 'answer_start_speed',
 'prepared_start_distance',
 'arrive_order_cnt_90d',
 'arrive_order_cnt_60d',
 'arrive_order_cnt_30d',
 'arrive_order_cnt_7d',
 'arrive_station_order_cnt_90d',
 'arrive_station_order_cnt_60d',
 'arrive_station_order_cnt_30d',
 'arrive_station_order_cnt_7d',
 'arrive_cross_order_cnt_90d',
 'arrive_cross_order_cnt_60d',
 'arrive_cross_order_cnt_30d',
 'arrive_cross_order_cnt_7d',
 'arrive_carp_succ_order_cnt_90d',
 'arrive_carp_succ_order_cnt_60d',
 'arrive_carp_succ_order_cnt_30d',
 'arrive_carp_succ_order_cnt_7d',
 'arrive_free_order_cnt_90d',
 'arrive_free_order_cnt_60d',
 'arrive_free_order_cnt_30d',
 'arrive_free_order_cnt_7d',
 'arrive_report_order_cnt_90d',
 'arrive_report_order_cnt_60d',
 'arrive_report_order_cnt_30d',
 'arrive_report_order_cnt_7d',
 'arrive_dis50_order_cnt_90d',
 'arrive_dis50_order_cnt_60d',
 'arrive_dis50_order_cnt_30d',
 'arrive_dis50_order_cnt_7d',
 'arrive_dis100_order_cnt_90d',
 'arrive_dis100_order_cnt_60d',
 'arrive_dis100_order_cnt_30d',
 'arrive_dis100_order_cnt_7d',
 'arrive_night_order_cnt_90d',
 'arrive_night_order_cnt_60d',
 'arrive_night_order_cnt_30d',
 'arrive_night_order_cnt_7d',
 'arrive_dis50_night_order_cnt_90d',
 'arrive_dis50_night_order_cnt_60d',
 'arrive_dis50_night_order_cnt_30d',
 'arrive_dis50_night_order_cnt_7d',
 'arrive_start_distance_1km_90d',
 'arrive_start_distance_1km_60d',
 'arrive_start_distance_1km_30d',
 'arrive_start_distance_1km_7d',
 'finish_dest_distance_1km_90d',
 'finish_dest_distance_1km_60d',
 'finish_dest_distance_1km_30d',
 'finish_dest_distance_1km_7d',
 'arrive_good_tag_order_cnt_90d',
 'arrive_good_tag_order_cnt_60d',
 'arrive_good_tag_order_cnt_30d',
 'arrive_good_tag_order_cnt_7d',
 'arrive_bad_tag_order_cnt_90d',
 'arrive_bad_tag_order_cnt_60d',
 'arrive_bad_tag_order_cnt_30d',
 'arrive_bad_tag_order_cnt_7d',
 'arrive_undefine_pas_order_cnt_90d',
 'arrive_undefine_pas_order_cnt_60d',
 'arrive_undefine_pas_order_cnt_30d',
 'arrive_undefine_pas_order_cnt_7d',
 'arrive_person_car_complaint_cut_90d',
 'arrive_person_car_complaint_cut_60d',
 'arrive_person_car_complaint_cut_30d',
 'arrive_person_car_complaint_cut_7d',
 'arrive_person_car_complaint_cut_real_90d',
 'arrive_person_car_complaint_cut_real_60d',
 'arrive_person_car_complaint_cut_real_30d',
 'arrive_person_car_complaint_cut_real_7d',
 'arrive_person_complaint_cut_90d',
 'arrive_person_complaint_cut_60d',
 'arrive_person_complaint_cut_30d',
 'arrive_person_complaint_cut_7d',
 'arrive_person_complaint_cut_real_90d',
 'arrive_person_complaint_cut_real_60d',
 'arrive_person_complaint_cut_real_30d',
 'arrive_person_complaint_cut_real_7d',
 'arrive_car_complaint_cut_90d',
 'arrive_car_complaint_cut_60d',
 'arrive_car_complaint_cut_30d',
 'arrive_car_complaint_cut_7d',
 'arrive_car_complaint_cut_real_90d',
 'arrive_car_complaint_cut_real_60d',
 'arrive_car_complaint_cut_real_30d',
 'arrive_car_complaint_cut_real_7d',
 'arrive_person_complaint_90d',
 'arrive_person_complaint_60d',
 'arrive_person_complaint_30d',
 'arrive_person_complaint_7d',
 'arrive_person_complaint_real_90d',
 'arrive_person_complaint_real_60d',
 'arrive_person_complaint_real_30d',
 'arrive_person_complaint_real_7d',
 'arrive_person_cut_90d',
 'arrive_person_cut_60d',
 'arrive_person_cut_30d',
 'arrive_person_cut_7d',
 'arrive_person_cut_real_90d',
 'arrive_person_cut_real_60d',
 'arrive_person_cut_real_30d',
 'arrive_person_cut_real_7d',
 'arrive_car_complaint_90d',
 'arrive_car_complaint_60d',
 'arrive_car_complaint_30d',
 'arrive_car_complaint_7d',
 'arrive_car_complaint_real_90d',
 'arrive_car_complaint_real_60d',
 'arrive_car_complaint_real_30d',
 'arrive_car_complaint_real_7d',
 'arrive_car_cut_90d',
 'arrive_car_cut_60d',
 'arrive_car_cut_30d',
 'arrive_car_cut_7d',
 'arrive_car_cut_real_90d',
 'arrive_car_cut_real_60d',
 'arrive_car_cut_real_30d',
 'arrive_car_cut_real_7d',
 'arrive_cond_plateno_not_same_90d',
 'arrive_cond_plateno_not_same_60d',
 'arrive_cond_plateno_not_same_30d',
 'arrive_cond_plateno_not_same_7d',
 'arrive_cond_brand_not_same_90d',
 'arrive_cond_brand_not_same_60d',
 'arrive_cond_brand_not_same_30d',
 'arrive_cond_brand_not_same_7d',
 'arrive_cond_series_not_same_90d',
 'arrive_cond_series_not_same_60d',
 'arrive_cond_series_not_same_30d',
 'arrive_cond_series_not_same_7d',
 'arrive_cond_change_car_90d',
 'arrive_cond_change_car_60d',
 'arrive_cond_change_car_30d',
 'arrive_cond_change_car_7d',
 'arrive_cond_color_not_same_90d',
 'arrive_cond_color_not_same_60d',
 'arrive_cond_color_not_same_30d',
 'arrive_cond_color_not_same_7d',
 'arrive_re_rule_order_cnt_90d',
 'arrive_re_rule_order_cnt_60d',
 'arrive_re_rule_order_cnt_30d',
 'arrive_re_rule_order_cnt_7d',
 'arrive_station_order_rate_90d',
 'arrive_station_order_rate_60d',
 'arrive_station_order_rate_30d',
 'arrive_station_order_rate_7d',
 'arrive_cross_order_rate_90d',
 'arrive_cross_order_rate_60d',
 'arrive_cross_order_rate_30d',
 'arrive_cross_order_rate_7d',
 'arrive_carp_succ_order_rate_90d',
 'arrive_carp_succ_order_rate_60d',
 'arrive_carp_succ_order_rate_30d',
 'arrive_carp_succ_order_rate_7d',
 'arrive_free_order_rate_90d',
 'arrive_free_order_rate_60d',
 'arrive_free_order_rate_30d',
 'arrive_free_order_rate_7d',
 'arrive_report_order_rate_90d',
 'arrive_report_order_rate_60d',
 'arrive_report_order_rate_30d',
 'arrive_report_order_rate_7d',
 'arrive_dis50_order_rate_90d',
 'arrive_dis50_order_rate_60d',
 'arrive_dis50_order_rate_30d',
 'arrive_dis50_order_rate_7d',
 'arrive_dis100_order_rate_90d',
 'arrive_dis100_order_rate_60d',
 'arrive_dis100_order_rate_30d',
 'arrive_dis100_order_rate_7d',
 'arrive_night_order_rate_90d',
 'arrive_night_order_rate_60d',
 'arrive_night_order_rate_30d',
 'arrive_night_order_rate_7d',
 'arrive_dis50_night_order_rate_90d',
 'arrive_dis50_night_order_rate_60d',
 'arrive_dis50_night_order_rate_30d',
 'arrive_dis50_night_order_rate_7d',
 'arrive_start_distance_1km_rate_90d',
 'arrive_start_distance_1km_rate_60d',
 'arrive_start_distance_1km_rate_30d',
 'arrive_start_distance_1km_rate_7d',
 'finish_dest_distance_1km_rate_90d',
 'finish_dest_distance_1km_rate_60d',
 'finish_dest_distance_1km_rate_30d',
 'finish_dest_distance_1km_rate_7d',
 'arrive_good_tag_order_rate_90d',
 'arrive_good_tag_order_rate_60d',
 'arrive_good_tag_order_rate_30d',
 'arrive_good_tag_order_rate_7d',
 'arrive_bad_tag_order_rate_90d',
 'arrive_bad_tag_order_rate_60d',
 'arrive_bad_tag_order_rate_30d',
 'arrive_bad_tag_order_rate_7d',
 'arrive_undefine_pas_order_rate_90d',
 'arrive_undefine_pas_order_rate_60d',
 'arrive_undefine_pas_order_rate_30d',
 'arrive_undefine_pas_order_rate_7d',
 'arrive_person_car_complaint_cut_rate_90d',
 'arrive_person_car_complaint_cut_rate_60d',
 'arrive_person_car_complaint_cut_rate_30d',
 'arrive_person_car_complaint_cut_rate_7d',
 'arrive_person_car_complaint_cut_real_rate_90d',
 'arrive_person_car_complaint_cut_real_rate_60d',
 'arrive_person_car_complaint_cut_real_rate_30d',
 'arrive_person_car_complaint_cut_real_rate_7d',
 'arrive_person_complaint_cut_rate_90d',
 'arrive_person_complaint_cut_rate_60d',
 'arrive_person_complaint_cut_rate_30d',
 'arrive_person_complaint_cut_rate_7d',
 'arrive_person_complaint_cut_real_rate_90d',
 'arrive_person_complaint_cut_real_rate_60d',
 'arrive_person_complaint_cut_real_rate_30d',
 'arrive_person_complaint_cut_real_rate_7d',
 'arrive_car_complaint_cut_rate_90d',
 'arrive_car_complaint_cut_rate_60d',
 'arrive_car_complaint_cut_rate_30d',
 'arrive_car_complaint_cut_rate_7d',
 'arrive_car_complaint_cut_real_rate_90d',
 'arrive_car_complaint_cut_real_rate_60d',
 'arrive_car_complaint_cut_real_rate_30d',
 'arrive_car_complaint_cut_real_rate_7d',
 'arrive_person_complaint_rate_90d',
 'arrive_person_complaint_rate_60d',
 'arrive_person_complaint_rate_30d',
 'arrive_person_complaint_rate_7d',
 'arrive_person_complaint_real_rate_90d',
 'arrive_person_complaint_real_rate_60d',
 'arrive_person_complaint_real_rate_30d',
 'arrive_person_complaint_real_rate_7d',
 'arrive_person_cut_rate_90d',
 'arrive_person_cut_rate_60d',
 'arrive_person_cut_rate_30d',
 'arrive_person_cut_rate_7d',
 'arrive_person_cut_real_rate_90d',
 'arrive_person_cut_real_rate_60d',
 'arrive_person_cut_real_rate_30d',
 'arrive_person_cut_real_rate_7d',
 'arrive_car_complaint_rate_90d',
 'arrive_car_complaint_rate_60d',
 'arrive_car_complaint_rate_30d',
 'arrive_car_complaint_rate_7d',
 'arrive_car_complaint_real_rate_90d',
 'arrive_car_complaint_real_rate_60d',
 'arrive_car_complaint_real_rate_30d',
 'arrive_car_complaint_real_rate_7d',
 'arrive_car_cut_rate_90d',
 'arrive_car_cut_rate_60d',
 'arrive_car_cut_rate_30d',
 'arrive_car_cut_rate_7d',
 'arrive_car_cut_real_rate_90d',
 'arrive_car_cut_real_rate_60d',
 'arrive_car_cut_real_rate_30d',
 'arrive_car_cut_real_rate_7d',
 'arrive_cond_plateno_not_same_rate_90d',
 'arrive_cond_plateno_not_same_rate_60d',
 'arrive_cond_plateno_not_same_rate_30d',
 'arrive_cond_plateno_not_same_rate_7d',
 'arrive_cond_brand_not_same_rate_90d',
 'arrive_cond_brand_not_same_rate_60d',
 'arrive_cond_brand_not_same_rate_30d',
 'arrive_cond_brand_not_same_rate_7d',
 'arrive_cond_series_not_same_rate_90d',
 'arrive_cond_series_not_same_rate_60d',
 'arrive_cond_series_not_same_rate_30d',
 'arrive_cond_series_not_same_rate_7d',
 'arrive_cond_change_car_rate_90d',
 'arrive_cond_change_car_rate_60d',
 'arrive_cond_change_car_rate_30d',
 'arrive_cond_change_car_rate_7d',
 'arrive_cond_color_not_same_rate_90d',
 'arrive_cond_color_not_same_rate_60d',
 'arrive_cond_color_not_same_rate_30d',
 'arrive_cond_color_not_same_rate_7d',
 'arrive_re_rule_order_rate_90d',
 'arrive_re_rule_order_rate_60d',
 'arrive_re_rule_order_rate_30d',
 'arrive_re_rule_order_rate_7d',

 'reg_dt_max',
 'reg_dt_min',
 'age_max',
 'age_min',
 'dishonest_status',
 'black_flag',
 '3m_route_time_type_max',
 'week_dri_flag',
 'is_profession',
 '15d_offline_flag',
 '15d_cancel_cash_flag',
 '30_high_cancel_sta_flag',
 'resident_reg_sample_city_max',
 'resident_reg_sample_city_min',
 'is_gulf_driver',
 'has_emergency',
 'emergency_is_sfc',
 'emergency_is_wyc',
 'car_cnt_sum',
 'car_age_max',
 'car_age_min',
 'car_price_max',
 'car_price_min',
 'car_city_contains_resident_reg',
 'car_city_not_contains_resident_reg',
 'travel_license_contains_card',
 'is_valid_status_sum',
 'travel_auth_state_min',
 'person_car_auth_state_min',
 'blacked_cnt',
 'blacked_woman_cnt',

 'answer_order_cnt',
 'cancel_order_cnt',
 'driver_cancel_order_cnt',
 'pas_cancel_order_cnt',
 'cancel_rate',
 'driver_cancel_rate',
 'pas_cancel_rate']]

    raw_df = raw_df[(raw_df.plate_no.isnull()==False)&(raw_df.brand_desc.isnull()==False)&(raw_df.color.isnull()==False)]
    raw_df['mandarine'] = raw_df.plate_no.apply(lambda x:1 if _mandarine_area(x) else 0)
    raw_records = raw_df.to_dict('records')
    new_records = split_list(raw_records)
    feature_records = list()
    for records in tqdm(new_records):
        gc.collect()
        t1 = time.time()
        print(datetime.now())
        tmp_list = _map_feature_multithread(records)
        print(datetime.now())
        t2 = time.time()
        print(t2-t1, 's')
        if tmp_list:
            feature_records += tmp_list
    # new_records = split_list(feature_records)
    # feature_records = list()
    # for records in tqdm(new_records):
    #     gc.collect()
    #     t1 = time.time()
    #     print(datetime.now())
    #     tmp_list = _map_segment_multithread(records)
    #     print(datetime.now())
    #     t2 = time.time()
    #     print(t2-t1, 's')
    #     feature_records += tmp_list
    dev_df = pd.DataFrame.from_records(feature_records)
    dev_df['label']= dev_df['label'].apply(lambda x: _format_label(x))
    dev_df.to_csv(output_path, encoding='utf8', index=False)
    return dev_df

def _map_feature(raw_data):
    try:
        status, fmap = cnf.get_features_multithread(raw_data)
    # print(fmap)
    except Exception as e:
        print(e)
        print(raw_data)
        return None

    if status:
        raw_data.update(fmap['features'])
        return raw_data
    else:
        # print(status, fmap)
        return None

def split_list(l1):
    new_list = list()
    ll = len(l1)
    base_num = 2000
    mod_v = ll % base_num
    shang_v = int(ll / base_num)
    r = 0
    for i in range(shang_v):
        l = i*base_num+1
        r = (i+1)*base_num
        tmp = l1[l:r]
        new_list.append(tmp)
    l = r+1
    tmp = l1[l:]
    new_list.append(tmp)
    return new_list

def _map_feature_multithread(raw_records):
    # result_list = list()
    # def get_result(res):
    #     result_list.append(res)
    rlen = len(raw_records)
    wait_time = 45 * (rlen / 2000) if rlen > 2000 else 45
    pool_cnt = multiprocessing.cpu_count()-1
    # for r in raw_records:
    #     t = _map_feature(r)
    with ProcessPool(pool_cnt) as pool:
        # result_list=pool.map(_map_feature,raw_records)
        # result_list=pool.imap(_map_feature,raw_records)
        # res = pool.map_async(_map_feature, raw_records, callback=get_result)
        result_list = pool.map_async(_map_feature, raw_records)
        result_list.wait(wait_time)
        try:
            result_list = result_list.get(wait_time)
            result_list = [t for t in result_list if t]
        except TimeoutError as e:
            print(e)
            result_list = []
 #         result_list = pool.map(_map_feature, raw_records)
#         pool.close()
#         pool.join()
    return result_list

def _mandarine_area(plate_no):
    first_char = plate_no[0]
    char_2nd = plate_no[1]
    fp = first_char in ['京', '沪', '津', '黑', '吉', '辽', '冀']
    sp = char_2nd in ['A', 'B', 'C']
    if fp or sp:
        return True
    else:
        return False

def _segment_contents(record):
    cont = record['contents']
    res = fake_p.pre_seg_contents_for_lime(cont)
    record['seg'] = res
    return record

def _map_segment_multithread(raw_records):
    # result_list = list()    pool = ThreadPool(2)
    pool_cnt = multiprocessing.cpu_count()-1
    with ProcessPool(pool_cnt) as pool:
        result_list=list(pool.map(_segment_contents,raw_records))
 #         result_list = pool.map(_map_feature, raw_records)
#     pool.close()
#     pool.join()
    result_list = [tmp_re for tmp_re in result_list if tmp_re]
    return result_list

def make_fasttext_file(train_df, file_path):
    tmp = train_df[['label', 'seg']]
    tmp.to_csv(file_path, index=False, header=None, encoding='utf8', sep='\t')

def new_predict(classifier, contents):
    try:
        l = classifier.predict(contents, k=-1, threshold=0.5)
        return l[0][0]
    except:
        return '__label__0'
def new_predict_prob(classifier, contents):
    try:
        l = classifier.predict(contents, k=-1, threshold=0.5)
    except:
        return 0
    if l[0][0] == '__label__0':
        return 1- l[1][0]
    else:
        return l[1][0]

def predict_prob_dk(classifier, contents):
    l = classifier.predict(contents, k=-1, threshold=0.5)
    prob = list()
    if l[0][0] == '__label__0':
        prob.append([l[1][0], 1 - l[1][0]])
    else:
        prob.append([1-l[1][0], l[1][0]])
    nd_prob = np.array(prob)
    return nd_prob

def fasttext_prediction_in_sklearn_format(classifier, texts):
    res = []
    # Ask FastText for the top 10 most likely labels for each piece of text.
    # This ensures we always get a probability score for every possible label in our model.
    labels, probabilities = classifier.predict(texts, 10)
#     print( labels, probabilities)
    # For each prediction, sort the probabaility scores into the same order
    # (I.e. no_stars, 1_star, 2_star, etc). This is needed because FastText
    # returns predicitons sorted by most likely instead of in a fixed order.
    for label, probs, text in zip(labels, probabilities, texts):
#         print(label, probs, text)
        order = np.argsort(np.array(label))
#         print(order)
        res.append(probs[order])
#         print(res)

    return np.array(res)

fake_p = FakerPreprocessor()
cnf = CarNlpFeature()

if __name__ == "__main__":
    datadir = '/nfs/volume-776-1/offline_data/'
    # datadir = '/Users/charleslai/projects/fake_model_sfc/offline_data'
    train_data_path = datadir + '/asr/trainset_20211209_all.txt'
    dev_data_path = datadir + '/asr/devset_20211209_all.txt'
    test_data_path = datadir + '/asr/testset_20211209_all.txt'
    generate_f_cols = [
    1010001000, 1010001001, 1010001002,
    1020001000, 1020001001, 1020001002, 1020001003, 1020001004, 1020001005, 1020001006, 1020001007, 1020001008, 1020001009, 1020001010, 1020001011, 1020001012, 1020001013, 1020001014, 1020001015, 1020001016, 1020001017, 1020001018, 1020001019, 1020001020, 1020001021, 1020001022,1020001023, 1020001024, 1020001025, 1020001026,
    1040001000, 1040001001, 1040001002, 1040001003, 1040001004, 1040001005, 1040001006, 1040001007, 1040001008, 1040001009, 1040001010, 1040001011, 1040001012, 1040001013, 1040001014, 1040001015, 1040001016, 1040001017, 1040001018, 1040001019, 1040001020, 1040001021, 1040001022,1040001023, 1040001024, 1040001025, 1040001026,
    1050001000, 1050001001, 1050001002, 1050001003, 1050001004, 1050001005, 1050001006, 1050001007, 1050001008, 1050001009, 1050001010, 1050001011, 1050001012, 1050001013, 1050001014, 1050001015, 1050001016, 1050001017, 1050001018, 1050001019, 1050001020, 1050001021, 1050001022,1050001023, 1050001024, 1050001025, 1050001026,]
    # standard_columns = ['final_gvid', 'd_pid', 'contents', 'plate_no', 'color', 'brand_desc', 'cur_date', 'label', 'mandarine']
    standard_columns = ['label',
 'o_id',
 'd_id',
 'contents',
 'plate_no',
 'color',
 'brand_desc',
 'cur_date',
 'depart_hour',
 'depart_weekday',
 'est_dis',
 'busi_type',
 'is_cross_city_flag',
 'is_station_flag',
 'is_extra_info',
 'is_luggage',
 'passenger_num',
 'd_gender',
 'p_gender',
 'man_driver_woman_pas',
 'seat_cnt',
 'friends_cnt',
 'depart_invite_dt',
 'answer_start_distance',
 'answer_start_speed',
 'prepared_start_distance',
 'arrive_order_cnt_90d',
 'arrive_order_cnt_60d',
 'arrive_order_cnt_30d',
 'arrive_order_cnt_7d',
 'arrive_station_order_cnt_90d',
 'arrive_station_order_cnt_60d',
 'arrive_station_order_cnt_30d',
 'arrive_station_order_cnt_7d',
 'arrive_cross_order_cnt_90d',
 'arrive_cross_order_cnt_60d',
 'arrive_cross_order_cnt_30d',
 'arrive_cross_order_cnt_7d',
 'arrive_carp_succ_order_cnt_90d',
 'arrive_carp_succ_order_cnt_60d',
 'arrive_carp_succ_order_cnt_30d',
 'arrive_carp_succ_order_cnt_7d',
 'arrive_free_order_cnt_90d',
 'arrive_free_order_cnt_60d',
 'arrive_free_order_cnt_30d',
 'arrive_free_order_cnt_7d',
 'arrive_report_order_cnt_90d',
 'arrive_report_order_cnt_60d',
 'arrive_report_order_cnt_30d',
 'arrive_report_order_cnt_7d',
 'arrive_dis50_order_cnt_90d',
 'arrive_dis50_order_cnt_60d',
 'arrive_dis50_order_cnt_30d',
 'arrive_dis50_order_cnt_7d',
 'arrive_dis100_order_cnt_90d',
 'arrive_dis100_order_cnt_60d',
 'arrive_dis100_order_cnt_30d',
 'arrive_dis100_order_cnt_7d',
 'arrive_night_order_cnt_90d',
 'arrive_night_order_cnt_60d',
 'arrive_night_order_cnt_30d',
 'arrive_night_order_cnt_7d',
 'arrive_dis50_night_order_cnt_90d',
 'arrive_dis50_night_order_cnt_60d',
 'arrive_dis50_night_order_cnt_30d',
 'arrive_dis50_night_order_cnt_7d',
 'arrive_start_distance_1km_90d',
 'arrive_start_distance_1km_60d',
 'arrive_start_distance_1km_30d',
 'arrive_start_distance_1km_7d',
 'finish_dest_distance_1km_90d',
 'finish_dest_distance_1km_60d',
 'finish_dest_distance_1km_30d',
 'finish_dest_distance_1km_7d',
 'arrive_good_tag_order_cnt_90d',
 'arrive_good_tag_order_cnt_60d',
 'arrive_good_tag_order_cnt_30d',
 'arrive_good_tag_order_cnt_7d',
 'arrive_bad_tag_order_cnt_90d',
 'arrive_bad_tag_order_cnt_60d',
 'arrive_bad_tag_order_cnt_30d',
 'arrive_bad_tag_order_cnt_7d',
 'arrive_undefine_pas_order_cnt_90d',
 'arrive_undefine_pas_order_cnt_60d',
 'arrive_undefine_pas_order_cnt_30d',
 'arrive_undefine_pas_order_cnt_7d',
 'arrive_person_car_complaint_cut_90d',
 'arrive_person_car_complaint_cut_60d',
 'arrive_person_car_complaint_cut_30d',
 'arrive_person_car_complaint_cut_7d',
 'arrive_person_car_complaint_cut_real_90d',
 'arrive_person_car_complaint_cut_real_60d',
 'arrive_person_car_complaint_cut_real_30d',
 'arrive_person_car_complaint_cut_real_7d',
 'arrive_person_complaint_cut_90d',
 'arrive_person_complaint_cut_60d',
 'arrive_person_complaint_cut_30d',
 'arrive_person_complaint_cut_7d',
 'arrive_person_complaint_cut_real_90d',
 'arrive_person_complaint_cut_real_60d',
 'arrive_person_complaint_cut_real_30d',
 'arrive_person_complaint_cut_real_7d',
 'arrive_car_complaint_cut_90d',
 'arrive_car_complaint_cut_60d',
 'arrive_car_complaint_cut_30d',
 'arrive_car_complaint_cut_7d',
 'arrive_car_complaint_cut_real_90d',
 'arrive_car_complaint_cut_real_60d',
 'arrive_car_complaint_cut_real_30d',
 'arrive_car_complaint_cut_real_7d',
 'arrive_person_complaint_90d',
 'arrive_person_complaint_60d',
 'arrive_person_complaint_30d',
 'arrive_person_complaint_7d',
 'arrive_person_complaint_real_90d',
 'arrive_person_complaint_real_60d',
 'arrive_person_complaint_real_30d',
 'arrive_person_complaint_real_7d',
 'arrive_person_cut_90d',
 'arrive_person_cut_60d',
 'arrive_person_cut_30d',
 'arrive_person_cut_7d',
 'arrive_person_cut_real_90d',
 'arrive_person_cut_real_60d',
 'arrive_person_cut_real_30d',
 'arrive_person_cut_real_7d',
 'arrive_car_complaint_90d',
 'arrive_car_complaint_60d',
 'arrive_car_complaint_30d',
 'arrive_car_complaint_7d',
 'arrive_car_complaint_real_90d',
 'arrive_car_complaint_real_60d',
 'arrive_car_complaint_real_30d',
 'arrive_car_complaint_real_7d',
 'arrive_car_cut_90d',
 'arrive_car_cut_60d',
 'arrive_car_cut_30d',
 'arrive_car_cut_7d',
 'arrive_car_cut_real_90d',
 'arrive_car_cut_real_60d',
 'arrive_car_cut_real_30d',
 'arrive_car_cut_real_7d',
 'arrive_cond_plateno_not_same_90d',
 'arrive_cond_plateno_not_same_60d',
 'arrive_cond_plateno_not_same_30d',
 'arrive_cond_plateno_not_same_7d',
 'arrive_cond_brand_not_same_90d',
 'arrive_cond_brand_not_same_60d',
 'arrive_cond_brand_not_same_30d',
 'arrive_cond_brand_not_same_7d',
 'arrive_cond_series_not_same_90d',
 'arrive_cond_series_not_same_60d',
 'arrive_cond_series_not_same_30d',
 'arrive_cond_series_not_same_7d',
 'arrive_cond_change_car_90d',
 'arrive_cond_change_car_60d',
 'arrive_cond_change_car_30d',
 'arrive_cond_change_car_7d',
 'arrive_cond_color_not_same_90d',
 'arrive_cond_color_not_same_60d',
 'arrive_cond_color_not_same_30d',
 'arrive_cond_color_not_same_7d',
 'arrive_re_rule_order_cnt_90d',
 'arrive_re_rule_order_cnt_60d',
 'arrive_re_rule_order_cnt_30d',
 'arrive_re_rule_order_cnt_7d',
 'arrive_station_order_rate_90d',
 'arrive_station_order_rate_60d',
 'arrive_station_order_rate_30d',
 'arrive_station_order_rate_7d',
 'arrive_cross_order_rate_90d',
 'arrive_cross_order_rate_60d',
 'arrive_cross_order_rate_30d',
 'arrive_cross_order_rate_7d',
 'arrive_carp_succ_order_rate_90d',
 'arrive_carp_succ_order_rate_60d',
 'arrive_carp_succ_order_rate_30d',
 'arrive_carp_succ_order_rate_7d',
 'arrive_free_order_rate_90d',
 'arrive_free_order_rate_60d',
 'arrive_free_order_rate_30d',
 'arrive_free_order_rate_7d',
 'arrive_report_order_rate_90d',
 'arrive_report_order_rate_60d',
 'arrive_report_order_rate_30d',
 'arrive_report_order_rate_7d',
 'arrive_dis50_order_rate_90d',
 'arrive_dis50_order_rate_60d',
 'arrive_dis50_order_rate_30d',
 'arrive_dis50_order_rate_7d',
 'arrive_dis100_order_rate_90d',
 'arrive_dis100_order_rate_60d',
 'arrive_dis100_order_rate_30d',
 'arrive_dis100_order_rate_7d',
 'arrive_night_order_rate_90d',
 'arrive_night_order_rate_60d',
 'arrive_night_order_rate_30d',
 'arrive_night_order_rate_7d',
 'arrive_dis50_night_order_rate_90d',
 'arrive_dis50_night_order_rate_60d',
 'arrive_dis50_night_order_rate_30d',
 'arrive_dis50_night_order_rate_7d',
 'arrive_start_distance_1km_rate_90d',
 'arrive_start_distance_1km_rate_60d',
 'arrive_start_distance_1km_rate_30d',
 'arrive_start_distance_1km_rate_7d',
 'finish_dest_distance_1km_rate_90d',
 'finish_dest_distance_1km_rate_60d',
 'finish_dest_distance_1km_rate_30d',
 'finish_dest_distance_1km_rate_7d',
 'arrive_good_tag_order_rate_90d',
 'arrive_good_tag_order_rate_60d',
 'arrive_good_tag_order_rate_30d',
 'arrive_good_tag_order_rate_7d',
 'arrive_bad_tag_order_rate_90d',
 'arrive_bad_tag_order_rate_60d',
 'arrive_bad_tag_order_rate_30d',
 'arrive_bad_tag_order_rate_7d',
 'arrive_undefine_pas_order_rate_90d',
 'arrive_undefine_pas_order_rate_60d',
 'arrive_undefine_pas_order_rate_30d',
 'arrive_undefine_pas_order_rate_7d',
 'arrive_person_car_complaint_cut_rate_90d',
 'arrive_person_car_complaint_cut_rate_60d',
 'arrive_person_car_complaint_cut_rate_30d',
 'arrive_person_car_complaint_cut_rate_7d',
 'arrive_person_car_complaint_cut_real_rate_90d',
 'arrive_person_car_complaint_cut_real_rate_60d',
 'arrive_person_car_complaint_cut_real_rate_30d',
 'arrive_person_car_complaint_cut_real_rate_7d',
 'arrive_person_complaint_cut_rate_90d',
 'arrive_person_complaint_cut_rate_60d',
 'arrive_person_complaint_cut_rate_30d',
 'arrive_person_complaint_cut_rate_7d',
 'arrive_person_complaint_cut_real_rate_90d',
 'arrive_person_complaint_cut_real_rate_60d',
 'arrive_person_complaint_cut_real_rate_30d',
 'arrive_person_complaint_cut_real_rate_7d',
 'arrive_car_complaint_cut_rate_90d',
 'arrive_car_complaint_cut_rate_60d',
 'arrive_car_complaint_cut_rate_30d',
 'arrive_car_complaint_cut_rate_7d',
 'arrive_car_complaint_cut_real_rate_90d',
 'arrive_car_complaint_cut_real_rate_60d',
 'arrive_car_complaint_cut_real_rate_30d',
 'arrive_car_complaint_cut_real_rate_7d',
 'arrive_person_complaint_rate_90d',
 'arrive_person_complaint_rate_60d',
 'arrive_person_complaint_rate_30d',
 'arrive_person_complaint_rate_7d',
 'arrive_person_complaint_real_rate_90d',
 'arrive_person_complaint_real_rate_60d',
 'arrive_person_complaint_real_rate_30d',
 'arrive_person_complaint_real_rate_7d',
 'arrive_person_cut_rate_90d',
 'arrive_person_cut_rate_60d',
 'arrive_person_cut_rate_30d',
 'arrive_person_cut_rate_7d',
 'arrive_person_cut_real_rate_90d',
 'arrive_person_cut_real_rate_60d',
 'arrive_person_cut_real_rate_30d',
 'arrive_person_cut_real_rate_7d',
 'arrive_car_complaint_rate_90d',
 'arrive_car_complaint_rate_60d',
 'arrive_car_complaint_rate_30d',
 'arrive_car_complaint_rate_7d',
 'arrive_car_complaint_real_rate_90d',
 'arrive_car_complaint_real_rate_60d',
 'arrive_car_complaint_real_rate_30d',
 'arrive_car_complaint_real_rate_7d',
 'arrive_car_cut_rate_90d',
 'arrive_car_cut_rate_60d',
 'arrive_car_cut_rate_30d',
 'arrive_car_cut_rate_7d',
 'arrive_car_cut_real_rate_90d',
 'arrive_car_cut_real_rate_60d',
 'arrive_car_cut_real_rate_30d',
 'arrive_car_cut_real_rate_7d',
 'arrive_cond_plateno_not_same_rate_90d',
 'arrive_cond_plateno_not_same_rate_60d',
 'arrive_cond_plateno_not_same_rate_30d',
 'arrive_cond_plateno_not_same_rate_7d',
 'arrive_cond_brand_not_same_rate_90d',
 'arrive_cond_brand_not_same_rate_60d',
 'arrive_cond_brand_not_same_rate_30d',
 'arrive_cond_brand_not_same_rate_7d',
 'arrive_cond_series_not_same_rate_90d',
 'arrive_cond_series_not_same_rate_60d',
 'arrive_cond_series_not_same_rate_30d',
 'arrive_cond_series_not_same_rate_7d',
 'arrive_cond_change_car_rate_90d',
 'arrive_cond_change_car_rate_60d',
 'arrive_cond_change_car_rate_30d',
 'arrive_cond_change_car_rate_7d',
 'arrive_cond_color_not_same_rate_90d',
 'arrive_cond_color_not_same_rate_60d',
 'arrive_cond_color_not_same_rate_30d',
 'arrive_cond_color_not_same_rate_7d',
 'arrive_re_rule_order_rate_90d',
 'arrive_re_rule_order_rate_60d',
 'arrive_re_rule_order_rate_30d',
 'arrive_re_rule_order_rate_7d',

 'reg_dt_max',
 'reg_dt_min',
 'age_max',
 'age_min',
 'dishonest_status',
 'black_flag',
 '3m_route_time_type_max',
 'week_dri_flag',
 'is_profession',
 '15d_offline_flag',
 '15d_cancel_cash_flag',
 '30_high_cancel_sta_flag',
 'resident_reg_sample_city_max',
 'resident_reg_sample_city_min',
 'is_gulf_driver',
 'has_emergency',
 'emergency_is_sfc',
 'emergency_is_wyc',
 'car_cnt_sum',
 'car_age_max',
 'car_age_min',
 'car_price_max',
 'car_price_min',
 'car_city_contains_resident_reg',
 'car_city_not_contains_resident_reg',
 'travel_license_contains_card',
 'is_valid_status_sum',
 'travel_auth_state_min',
 'person_car_auth_state_min',
 'blacked_cnt',
 'blacked_woman_cnt',

 'answer_order_cnt',
 'cancel_order_cnt',
 'driver_cancel_order_cnt',
 'pas_cancel_order_cnt',
 'cancel_rate',
 'driver_cancel_rate',
 'pas_cancel_rate','mandarine']
    standard_columns = standard_columns + generate_f_cols + ['seg']
    train_df = preprocess_data_set(train_data_path, datadir + '/asr/data/train_set_person_df_20211209_all.csv')
    # train_df = pd.read_csv(datadir + '/asr/data/train_set_person_df_20211011.csv')
    train_df.columns = standard_columns
    train_df = train_df.replace({'seg': {'DIGIT': ''}}, regex=True)
    dev_df = preprocess_data_set(dev_data_path, datadir + '/asr/data/dev_set_person_df_20211209_all.csv')
    # dev_df = pd.read_csv(datadir + '/asr/data/dev_set_person_df_20211011.csv')
    dev_df.columns = standard_columns
    dev_df = dev_df.replace({'seg': {'DIGIT': ''}}, regex=True)
    test_df = preprocess_data_set(test_data_path, datadir + '/asr/data/test_set_person_df_20211209_all.csv')
    # test_df = pd.read_csv(datadir + '/asr/data/train_set_person_df_20211011.csv')
    test_df.columns = standard_columns
    test_df = test_df.replace({'seg': {'DIGIT': ''}}, regex=True)

#     # train_df = preprocess_data_set(train_data_path, datadir + '/im/data/train_set_df_20210305.csv')
#     # dev_df = preprocess_data_set(dev_data_path, datadir + '/im/data/dev_set_df_20210305.csv')
#     # test_df = preprocess_data_set(test_data_path, datadir + '/im/data/test_set_df_20210305.csv')ll
#     # test_df = pd.read_csv(datadir + '/im/data/test_set_df_20210305.csv')
    train_df['label']= train_df['label'].apply(lambda x: _format_label(x))
    train_df['label_i'] =  train_df['label'].apply(lambda x: _re_label_int(x))
    dev_df['label']= dev_df['label'].apply(lambda x: _format_label(x))
    dev_df['label_i'] =  dev_df['label'].apply(lambda x: _re_label_int(x))
    train_file = train_data_path + '.input'
    dev_file = dev_data_path + '.input'
    make_fasttext_file(train_df, train_file)
    make_fasttext_file(dev_df, dev_file)
    test_file = test_data_path + '.input'
    make_fasttext_file(test_df, test_file)
#     TODO Better fasttext
#     more epochs and larger learning rate
#     classifier = fasttext.train_supervised(input=train_file,
#                                        autotuneValidationFile=dev_file,
#                                        lr=0.25, epoch=50,
#                                        wordNgrams=3,
#                                        # minCount=10,minn=1,
#                                        bucket=2000000, dim=256, loss='hs',
#                                        autotuneDuration=2000,
#                                        autotuneMetric="f1:__label__1")
#     classifier.save_model(datadir + '/model/asr_person_ft_v20211011_2.model')

    classifier = fasttext.load_model(datadir + '/model/asr_person_ft_v20211011_2.model')
    result = classifier.test(test_file)
    print(result)
    test_df['predict'] = test_df.seg.apply(lambda x:new_predict(classifier, x))
    test_df['prob'] = test_df.seg.apply(lambda x:new_predict_prob(classifier, x))
    test_df['label']= test_df['label'].apply(lambda x: _format_label(x))
    test_df['label_i'] =  test_df['label'].apply(lambda x: _re_label_int(x))

    report = classification_report(test_df.label, test_df.predict)
    print('v20211011_2')
    print(report)
    bce = BinaryClassEstimate()
    y_true = test_df['label_i']
    y_pred = test_df['prob']
    pred_res = [{'label':l, 'score':p} for l, p in zip(y_true, y_pred)]
    metric_param = {
        'train_result': pred_res,
        'proj_name': 'test',
    }

    mapped_y_pred = bce.mapping(y_pred=y_pred, mapping_model='grade', **metric_param)
    bce.count_score_grade(mapped_y_pred, y_true)
    bce.print_score_pred_equal_width_table(y_pred, y_true)

    # train_df['prob'] = test_df.seg.apply(lambda x:new_predict_prob(classifier, x))
    # dev_df['prob'] = test_df.seg.apply(lambda x:new_predict_prob(classifier, x))

    train_df['predict'] = train_df.seg.apply(lambda x:new_predict(classifier, x))
    train_df['prob'] = train_df.seg.apply(lambda x:new_predict_prob(classifier, x))
    train_df['label']= train_df['label'].apply(lambda x: _format_label(x))
    train_df['label_i'] =  train_df['label'].apply(lambda x: _re_label_int(x))

    dev_df['predict'] = dev_df.seg.apply(lambda x:new_predict(classifier, x))
    dev_df['prob'] = dev_df.seg.apply(lambda x:new_predict_prob(classifier, x))
    dev_df['label']= dev_df['label'].apply(lambda x: _format_label(x))
    dev_df['label_i'] =  dev_df['label'].apply(lambda x: _re_label_int(x))

    # # add_order_feas
    # order_feas = pd.read_csv('/nfs/volume-776-1/offline_data/asr/order_feas_0401_0906.txt', sep='\t',
    #                      dtype={'order_id': str, 'driver_id': str,'passenger_id':str,'route_id':str})
    # train_df = pd.merge(train_df,order_feas)
    # dev_df = pd.merge(dev_df,order_feas)
    # test_df = pd.merge(test_df,order_feas)

    # xgb
    generate_f_cols_filter = [
#     1010001000, 1010001001, 1010001002,
#     1020001000, 1020001001, 1020001002, 1020001003, 1020001004, 1020001005, 1020001006, 1020001007, 1020001008, 1020001009, 1020001010, 1020001011, 1020001012, 1020001013, 1020001014, 1020001015, 1020001016, 1020001017, 1020001018, 1020001019, 1020001020, 1020001021, 1020001022,1020001023,
#         1020001024,
#     1040001000, 1040001001, 1040001002, 1040001003, 1040001004, 1040001005, 1040001006, 1040001007, 1040001008, 1040001009, 1040001010, 1040001011, 1040001012, 1040001013, 1040001014, 1040001015, 1040001016, 1040001017, 1040001018, 1040001019, 1040001020, 1040001021, 1040001022,1040001023,
#         1040001024,
#     1050001000, 1050001001, 1050001002, 1050001003, 1050001004, 1050001005, 1050001006, 1050001007, 1050001008, 1050001009, 1050001010, 1050001011, 1050001012, 1050001013, 1050001014, 1050001015, 1050001016, 1050001017, 1050001018, 1050001019, 1050001020, 1050001021, 1050001022,1050001023,
        1050001024,
        1050001025,
        1050001026
        ]

#     order_feas_list = ['depart_hour', 'depart_weekday',
# #                        'est_dis',
# #        'busi_type', 'is_cross_city_flag',
#                        'is_station_flag',
#                        'is_extra_info', 'is_luggage', 'passenger_num',
# #                        'man_driver_woman_pas',
#                        'seat_cnt', 'friends_cnt', 'depart_invite_dt',
#        'answer_start_distance', 'answer_start_speed']


    order_feas_list = [
 'depart_hour',
 'depart_weekday',
 'est_dis',
 'busi_type',
 'is_cross_city_flag',
 'is_station_flag',
 'is_extra_info',
 'is_luggage',
 'passenger_num',
 'd_gender',
 'p_gender',
 'man_driver_woman_pas',
 'seat_cnt',
 'friends_cnt',
 'depart_invite_dt',
 'answer_start_distance',
 'answer_start_speed',
 'prepared_start_distance',
 'arrive_order_cnt_90d',
 'arrive_order_cnt_60d',
 'arrive_order_cnt_30d',
 'arrive_order_cnt_7d',
 'arrive_station_order_cnt_90d',
 'arrive_station_order_cnt_60d',
 'arrive_station_order_cnt_30d',
 'arrive_station_order_cnt_7d',
 'arrive_cross_order_cnt_90d',
 'arrive_cross_order_cnt_60d',
 'arrive_cross_order_cnt_30d',
 'arrive_cross_order_cnt_7d',
 'arrive_carp_succ_order_cnt_90d',
 'arrive_carp_succ_order_cnt_60d',
 'arrive_carp_succ_order_cnt_30d',
 'arrive_carp_succ_order_cnt_7d',
 'arrive_free_order_cnt_90d',
 'arrive_free_order_cnt_60d',
 'arrive_free_order_cnt_30d',
 'arrive_free_order_cnt_7d',
 'arrive_report_order_cnt_90d',
 'arrive_report_order_cnt_60d',
 'arrive_report_order_cnt_30d',
 'arrive_report_order_cnt_7d',
 'arrive_dis50_order_cnt_90d',
 'arrive_dis50_order_cnt_60d',
 'arrive_dis50_order_cnt_30d',
 'arrive_dis50_order_cnt_7d',
 'arrive_dis100_order_cnt_90d',
 'arrive_dis100_order_cnt_60d',
 'arrive_dis100_order_cnt_30d',
 'arrive_dis100_order_cnt_7d',
 'arrive_night_order_cnt_90d',
 'arrive_night_order_cnt_60d',
 'arrive_night_order_cnt_30d',
 'arrive_night_order_cnt_7d',
 'arrive_dis50_night_order_cnt_90d',
 'arrive_dis50_night_order_cnt_60d',
 'arrive_dis50_night_order_cnt_30d',
 'arrive_dis50_night_order_cnt_7d',
 'arrive_start_distance_1km_90d',
 'arrive_start_distance_1km_60d',
 'arrive_start_distance_1km_30d',
 'arrive_start_distance_1km_7d',
 'finish_dest_distance_1km_90d',
 'finish_dest_distance_1km_60d',
 'finish_dest_distance_1km_30d',
 'finish_dest_distance_1km_7d',
 'arrive_good_tag_order_cnt_90d',
 'arrive_good_tag_order_cnt_60d',
 'arrive_good_tag_order_cnt_30d',
 'arrive_good_tag_order_cnt_7d',
 'arrive_bad_tag_order_cnt_90d',
 'arrive_bad_tag_order_cnt_60d',
 'arrive_bad_tag_order_cnt_30d',
 'arrive_bad_tag_order_cnt_7d',
 'arrive_undefine_pas_order_cnt_90d',
 'arrive_undefine_pas_order_cnt_60d',
 'arrive_undefine_pas_order_cnt_30d',
 'arrive_undefine_pas_order_cnt_7d',
 'arrive_person_car_complaint_cut_90d',
 'arrive_person_car_complaint_cut_60d',
 'arrive_person_car_complaint_cut_30d',
 'arrive_person_car_complaint_cut_7d',
 'arrive_person_car_complaint_cut_real_90d',
 'arrive_person_car_complaint_cut_real_60d',
 'arrive_person_car_complaint_cut_real_30d',
 'arrive_person_car_complaint_cut_real_7d',
 'arrive_person_complaint_cut_90d',
 'arrive_person_complaint_cut_60d',
 'arrive_person_complaint_cut_30d',
 'arrive_person_complaint_cut_7d',
 'arrive_person_complaint_cut_real_90d',
 'arrive_person_complaint_cut_real_60d',
 'arrive_person_complaint_cut_real_30d',
 'arrive_person_complaint_cut_real_7d',
 'arrive_car_complaint_cut_90d',
 'arrive_car_complaint_cut_60d',
 'arrive_car_complaint_cut_30d',
 'arrive_car_complaint_cut_7d',
 'arrive_car_complaint_cut_real_90d',
 'arrive_car_complaint_cut_real_60d',
 'arrive_car_complaint_cut_real_30d',
 'arrive_car_complaint_cut_real_7d',
 'arrive_person_complaint_90d',
 'arrive_person_complaint_60d',
 'arrive_person_complaint_30d',
 'arrive_person_complaint_7d',
 'arrive_person_complaint_real_90d',
 'arrive_person_complaint_real_60d',
 'arrive_person_complaint_real_30d',
 'arrive_person_complaint_real_7d',
 'arrive_person_cut_90d',
 'arrive_person_cut_60d',
 'arrive_person_cut_30d',
 'arrive_person_cut_7d',
 'arrive_person_cut_real_90d',
 'arrive_person_cut_real_60d',
 'arrive_person_cut_real_30d',
 'arrive_person_cut_real_7d',
 'arrive_car_complaint_90d',
 'arrive_car_complaint_60d',
 'arrive_car_complaint_30d',
 'arrive_car_complaint_7d',
 'arrive_car_complaint_real_90d',
 'arrive_car_complaint_real_60d',
 'arrive_car_complaint_real_30d',
 'arrive_car_complaint_real_7d',
 'arrive_car_cut_90d',
 'arrive_car_cut_60d',
 'arrive_car_cut_30d',
 'arrive_car_cut_7d',
 'arrive_car_cut_real_90d',
 'arrive_car_cut_real_60d',
 'arrive_car_cut_real_30d',
 'arrive_car_cut_real_7d',
 'arrive_cond_plateno_not_same_90d',
 'arrive_cond_plateno_not_same_60d',
 'arrive_cond_plateno_not_same_30d',
 'arrive_cond_plateno_not_same_7d',
 'arrive_cond_brand_not_same_90d',
 'arrive_cond_brand_not_same_60d',
 'arrive_cond_brand_not_same_30d',
 'arrive_cond_brand_not_same_7d',
 'arrive_cond_series_not_same_90d',
 'arrive_cond_series_not_same_60d',
 'arrive_cond_series_not_same_30d',
 'arrive_cond_series_not_same_7d',
 'arrive_cond_change_car_90d',
 'arrive_cond_change_car_60d',
 'arrive_cond_change_car_30d',
 'arrive_cond_change_car_7d',
 'arrive_cond_color_not_same_90d',
 'arrive_cond_color_not_same_60d',
 'arrive_cond_color_not_same_30d',
 'arrive_cond_color_not_same_7d',
 'arrive_re_rule_order_cnt_90d',
 'arrive_re_rule_order_cnt_60d',
 'arrive_re_rule_order_cnt_30d',
 'arrive_re_rule_order_cnt_7d',
 'arrive_station_order_rate_90d',
 'arrive_station_order_rate_60d',
 'arrive_station_order_rate_30d',
 'arrive_station_order_rate_7d',
 'arrive_cross_order_rate_90d',
 'arrive_cross_order_rate_60d',
 'arrive_cross_order_rate_30d',
 'arrive_cross_order_rate_7d',
 'arrive_carp_succ_order_rate_90d',
 'arrive_carp_succ_order_rate_60d',
 'arrive_carp_succ_order_rate_30d',
 'arrive_carp_succ_order_rate_7d',
 'arrive_free_order_rate_90d',
 'arrive_free_order_rate_60d',
 'arrive_free_order_rate_30d',
 'arrive_free_order_rate_7d',
 'arrive_report_order_rate_90d',
 'arrive_report_order_rate_60d',
 'arrive_report_order_rate_30d',
 'arrive_report_order_rate_7d',
 'arrive_dis50_order_rate_90d',
 'arrive_dis50_order_rate_60d',
 'arrive_dis50_order_rate_30d',
 'arrive_dis50_order_rate_7d',
 'arrive_dis100_order_rate_90d',
 'arrive_dis100_order_rate_60d',
 'arrive_dis100_order_rate_30d',
 'arrive_dis100_order_rate_7d',
 'arrive_night_order_rate_90d',
 'arrive_night_order_rate_60d',
 'arrive_night_order_rate_30d',
 'arrive_night_order_rate_7d',
 'arrive_dis50_night_order_rate_90d',
 'arrive_dis50_night_order_rate_60d',
 'arrive_dis50_night_order_rate_30d',
 'arrive_dis50_night_order_rate_7d',
 'arrive_start_distance_1km_rate_90d',
 'arrive_start_distance_1km_rate_60d',
 'arrive_start_distance_1km_rate_30d',
 'arrive_start_distance_1km_rate_7d',
 'finish_dest_distance_1km_rate_90d',
 'finish_dest_distance_1km_rate_60d',
 'finish_dest_distance_1km_rate_30d',
 'finish_dest_distance_1km_rate_7d',
 'arrive_good_tag_order_rate_90d',
 'arrive_good_tag_order_rate_60d',
 'arrive_good_tag_order_rate_30d',
 'arrive_good_tag_order_rate_7d',
 'arrive_bad_tag_order_rate_90d',
 'arrive_bad_tag_order_rate_60d',
 'arrive_bad_tag_order_rate_30d',
 'arrive_bad_tag_order_rate_7d',
 'arrive_undefine_pas_order_rate_90d',
 'arrive_undefine_pas_order_rate_60d',
 'arrive_undefine_pas_order_rate_30d',
 'arrive_undefine_pas_order_rate_7d',
 'arrive_person_car_complaint_cut_rate_90d',
 'arrive_person_car_complaint_cut_rate_60d',
 'arrive_person_car_complaint_cut_rate_30d',
 'arrive_person_car_complaint_cut_rate_7d',
 'arrive_person_car_complaint_cut_real_rate_90d',
 'arrive_person_car_complaint_cut_real_rate_60d',
 'arrive_person_car_complaint_cut_real_rate_30d',
 'arrive_person_car_complaint_cut_real_rate_7d',
 'arrive_person_complaint_cut_rate_90d',
 'arrive_person_complaint_cut_rate_60d',
 'arrive_person_complaint_cut_rate_30d',
 'arrive_person_complaint_cut_rate_7d',
 'arrive_person_complaint_cut_real_rate_90d',
 'arrive_person_complaint_cut_real_rate_60d',
 'arrive_person_complaint_cut_real_rate_30d',
 'arrive_person_complaint_cut_real_rate_7d',
 'arrive_car_complaint_cut_rate_90d',
 'arrive_car_complaint_cut_rate_60d',
 'arrive_car_complaint_cut_rate_30d',
 'arrive_car_complaint_cut_rate_7d',
 'arrive_car_complaint_cut_real_rate_90d',
 'arrive_car_complaint_cut_real_rate_60d',
 'arrive_car_complaint_cut_real_rate_30d',
 'arrive_car_complaint_cut_real_rate_7d',
 'arrive_person_complaint_rate_90d',
 'arrive_person_complaint_rate_60d',
 'arrive_person_complaint_rate_30d',
 'arrive_person_complaint_rate_7d',
 'arrive_person_complaint_real_rate_90d',
 'arrive_person_complaint_real_rate_60d',
 'arrive_person_complaint_real_rate_30d',
 'arrive_person_complaint_real_rate_7d',
 'arrive_person_cut_rate_90d',
 'arrive_person_cut_rate_60d',
 'arrive_person_cut_rate_30d',
 'arrive_person_cut_rate_7d',
 'arrive_person_cut_real_rate_90d',
 'arrive_person_cut_real_rate_60d',
 'arrive_person_cut_real_rate_30d',
 'arrive_person_cut_real_rate_7d',
 'arrive_car_complaint_rate_90d',
 'arrive_car_complaint_rate_60d',
 'arrive_car_complaint_rate_30d',
 'arrive_car_complaint_rate_7d',
 'arrive_car_complaint_real_rate_90d',
 'arrive_car_complaint_real_rate_60d',
 'arrive_car_complaint_real_rate_30d',
 'arrive_car_complaint_real_rate_7d',
 'arrive_car_cut_rate_90d',
 'arrive_car_cut_rate_60d',
 'arrive_car_cut_rate_30d',
 'arrive_car_cut_rate_7d',
 'arrive_car_cut_real_rate_90d',
 'arrive_car_cut_real_rate_60d',
 'arrive_car_cut_real_rate_30d',
 'arrive_car_cut_real_rate_7d',
 'arrive_cond_plateno_not_same_rate_90d',
 'arrive_cond_plateno_not_same_rate_60d',
 'arrive_cond_plateno_not_same_rate_30d',
 'arrive_cond_plateno_not_same_rate_7d',
 'arrive_cond_brand_not_same_rate_90d',
 'arrive_cond_brand_not_same_rate_60d',
 'arrive_cond_brand_not_same_rate_30d',
 'arrive_cond_brand_not_same_rate_7d',
 'arrive_cond_series_not_same_rate_90d',
 'arrive_cond_series_not_same_rate_60d',
 'arrive_cond_series_not_same_rate_30d',
 'arrive_cond_series_not_same_rate_7d',
 'arrive_cond_change_car_rate_90d',
 'arrive_cond_change_car_rate_60d',
 'arrive_cond_change_car_rate_30d',
 'arrive_cond_change_car_rate_7d',
 'arrive_cond_color_not_same_rate_90d',
 'arrive_cond_color_not_same_rate_60d',
 'arrive_cond_color_not_same_rate_30d',
 'arrive_cond_color_not_same_rate_7d',
 'arrive_re_rule_order_rate_90d',
 'arrive_re_rule_order_rate_60d',
 'arrive_re_rule_order_rate_30d',
 'arrive_re_rule_order_rate_7d',

 'reg_dt_max',
 'reg_dt_min',
 'age_max',
 'age_min',
 'dishonest_status',
 'black_flag',
 '3m_route_time_type_max',
 'week_dri_flag',
 'is_profession',
 '15d_offline_flag',
 '15d_cancel_cash_flag',
 '30_high_cancel_sta_flag',
 'resident_reg_sample_city_max',
 'resident_reg_sample_city_min',
 'is_gulf_driver',
 'has_emergency',
 'emergency_is_sfc',
 'emergency_is_wyc',
 'car_cnt_sum',
 'car_age_max',
 'car_age_min',
 'car_price_max',
 'car_price_min',
 'car_city_contains_resident_reg',
 'car_city_not_contains_resident_reg',
 'travel_license_contains_card',
 'is_valid_status_sum',
 'travel_auth_state_min',
 'person_car_auth_state_min',
 'blacked_cnt',
 'blacked_woman_cnt',

 'answer_order_cnt',
 'cancel_order_cnt',
 'driver_cancel_order_cnt',
 'pas_cancel_order_cnt',
 'cancel_rate',
 'driver_cancel_rate',
 'pas_cancel_rate']

    f_cols = generate_f_cols_filter +['prob']+order_feas_list
    label_col = ['label_i']

    train_X = train_df[f_cols]
    train_Y = train_df[label_col]
    dev_X = dev_df[f_cols]
    dev_Y = dev_df[label_col]
    test_X = test_df[f_cols]
    test_Y = test_df[label_col]

    clf = xgb.XGBClassifier(
        booster='dart',
#         n_estimators=200,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        # missing=-1,
        eval_metric='logloss',
        scale_pos_weight=2,
        # max_delta_step=5,
        # USE CPU
#         nthread=4,
        tree_method='exact' ,
        objective='binary:logistic',
        sample_type='weighted',



    )
    clf.fit(train_X, train_Y.values.ravel(),
        eval_set=[(train_X, train_Y), (dev_X, dev_Y)],
        verbose=True)
    joblib.dump(clf, datadir + "/model/asr_person_history_ft_xgb_v20211209_1.model")


    evals_result = clf.evals_result()
    preds = clf.predict(test_X)
    probs = clf.predict_proba(test_X)[:, 1]
    test_df['predict2'] = preds
    test_df['probs'] = probs
#     report = classification_report(test_df.label_i, test_df.predict2)
#     print('v20211011+XGB_2')
#     print(report)
#     bce = BinaryClassEstimate()
    y_true = test_df['label_i']
    y_pred = test_df['probs']
    pred_res = [{'label':l, 'score':p} for l, p in zip(y_true, y_pred)]
    metric_param = {
        'train_result': pred_res,
        'proj_name': 'test',
    }

#     mapped_y_pred = bce.mapping(y_pred=y_pred, mapping_model='grade', **metric_param)
#     bce.count_score_grade(mapped_y_pred, y_true)
#     bce.print_score_pred_equal_width_table(y_pred, y_true)


    preds_train = clf.predict(train_X)
    probs_train = clf.predict_proba(train_X)[:, 1]
    train_df['predict2'] = preds_train
    train_df['probs'] = probs_train

    preds_dev = clf.predict(dev_X)
    probs_dev = clf.predict_proba(dev_X)[:, 1]
    dev_df['predict2'] = preds_dev
    dev_df['probs'] = probs_dev

    print(clf.feature_importances_)
    ax = xgb.plot_importance(clf)
    ax.figure.savefig('xgboost_1209_1.png')

    imp = pd.DataFrame({'importance':clf.feature_importances_, 'var':f_cols})
    imp = imp.sort_values(by="importance", ascending=False)
    print(imp)

    train_auc = roc_auc_score(train_df.label, train_df.predict2)
    dev_auc = roc_auc_score(dev_df.label, dev_df.predict2)
    test_auc = roc_auc_score(test_df.label, test_df.predict2)
    print("AUC--train--dev--test:", train_auc, dev_auc, test_auc)

    report = classification_report(test_df.label_i, test_df.predict2)
    print('v20211209+XGB_1')
    print(report)


    train_df.to_csv(datadir + '/predict/train_df_predict_20211209_all.csv')
    dev_df.to_csv(datadir + '/predict/dev_df_predict_20211209_all.csv')
    test_df.to_csv(datadir + '/predict/test_df_predict_20211209_all.csv')

    bce = BinaryClassEstimate()
    mapped_y_pred = bce.mapping(y_pred=y_pred, mapping_model='grade', **metric_param)
    bce.count_score_grade(mapped_y_pred, y_true)
    bce.print_score_pred_equal_width_table(y_pred, y_true)
