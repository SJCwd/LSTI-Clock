import os
import json
import random

import numpy as np
from easydict import EasyDict as edict


class MovieDataset(object):
    def __init__(self, data_dir,args):
        self.args = args
        self.noise = args.noise
        self.noise_rate = args.noise_rate

        self.data_dir = data_dir + '/Graph_generate_data'
        self.load_entities()
        self.load_relations()

    def get_relation(self):
        #Entities
        USER = 'user'
        ITEM = 'item'
        FEATURE = 'feature'

        #Relations  movie只有interact 和 belong_to
        INTERACT = 'interact'
        # FRIEND = 'friends'
        # LIKE = 'like'
        BELONG_TO = 'belong_to'

        # relation_name = [INTERACT, FRIEND, LIKE, BELONG_TO]
        relation_name = [INTERACT,BELONG_TO]

        myelp_relation = {
            USER: {
                INTERACT: ITEM,
                # FRIEND: USER,
                # LIKE: FEATURE,
            },
            ITEM: {
                BELONG_TO: FEATURE,
                INTERACT: USER
            },
            FEATURE: {
                # LIKE: USER,
                BELONG_TO: ITEM
            }
        }

        myelp_relation_link_entity_type = {
            INTERACT:  [USER, ITEM],
            # FRIEND:  [USER, USER],
            # LIKE:  [USER, FEATURE],
            BELONG_TO:  [ITEM, FEATURE]
        }
        return myelp_relation, relation_name, myelp_relation_link_entity_type

    # 设置属性 user item feature  值为一个edict(id = list , value_len = int)
    def load_entities(self):
        entity_files = edict(
            user='user_dict.json',
            item='item_dict.json',
            feature='tag_map.json',
        )
        for entity_name in entity_files:
            with open(os.path.join(self.data_dir,entity_files[entity_name]), encoding='utf-8') as f:
                mydict = json.load(f)
            if entity_name == 'feature':
                entity_id = list(mydict.values())   #编码后的
            else:   # user item
                entity_id = list(map(int, list(mydict.keys())))   #map函数是一个允许你使用另一个函数转换整个可迭代对象的函数 包括但不限于(字符串转为数字、四舍五入数字、获取每个可迭代项的长度)
            setattr(self, entity_name, edict(id=entity_id, value_len=max(entity_id)+1))  #将entity设置为属性（名称为：user、item、feature） 值为一个edict(id = list , value_len = int)
            print('Load', entity_name, 'of size', len(entity_id))
            print(entity_name, 'of max id is', max(entity_id))

    # 设置属性：movie只有  interact belong_to   没有friends like   值为一个edict(id = list , value_len = int)
    def load_relations(self):
        """
        relation: head entity---> tail entity
        --
        """
        Book_relations = edict(
            interact=('user_item_train.json', self.user, self.item), #(filename, head_entity, tail_entity)
            # friends=('user_dict.json', self.user, self.user),
            # like=('user_dict.json', self.user, self.feature),
            belong_to=('item_dict.json', self.item, self.feature),
        )

        #对每种relation都独立存储 将之作为类的attribute 值为一个edict(data = [[],...,[]])
        for name in Book_relations:  #interaction\friends\like\belong_to
            #  Save tail_entity
            relation = edict(
                data=[],
            )
            knowledge = [list([]) for i in range(Book_relations[name][1].value_len)]   #head有几个entity,就创建几个list(相当于是矩阵中的行)
            # load relation files
            with open(os.path.join(self.data_dir, Book_relations[name][0]), encoding='utf-8') as f:
                mydict = json.load(f)
            if name in ['interact']:
                for key, value in mydict.items():  # value是一个list,元素是字典
                    head_id = int(key)
                    # tail_ids = value  原本code的
                    tail_ids = [review['item'] for review in value]    #review属性：item,date,review_id   可能此处在读取数据时要把时间加上！！

                    ################# 数据污染 #############################
                    if self.noise:   #要进行污染
                        before_len = len(tail_ids)
                        noise_num = int(before_len * self.noise_rate)
                        if noise_num < 1:
                            noise_num = 1
                        # 随机选择
                        # tail_ids = tail_ids + random.sample(self.item.id, k=noise_num)

                        # 由添加改为替换  sample 是相当于不放回抽样  choices 是相当于放回抽样
                        noise_ids = random.sample(self.item.id, k=noise_num)
                        noise_index = random.sample(range(before_len), k=noise_num)
                        for i in range(noise_num):
                            tail_ids[noise_index[i]] = noise_ids[i]
                    ################# 数据污染 #############################
                    knowledge[head_id] = tail_ids

            elif name in ['friends', 'like']:  #user_dict.json  my_yelp没有like属性
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str][name]
                    knowledge[head_id] = tail_ids
            elif name in ['belong_to']:   #item_dict.json
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str]['feature_index']
                    knowledge[head_id] = tail_ids
            relation.data = knowledge

            setattr(self, name, relation)   #存储为类的属性
            tuple_num = 0
            for i in knowledge:
                tuple_num += len(i)
            print('Load', name, 'of size', tuple_num)   #该relation有几个记录






