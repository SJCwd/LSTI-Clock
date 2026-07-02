
class MovieGraph(object):

    def __init__(self, dataset,args):
        self.args = args
        self.noise = args.noise
        self.noise_rate = args.noise_rate

        self.G = dict()
        self._load_entities(dataset)
        self._load_knowledge(dataset)
        self._clean()

    # self.G[entity][eid] = {r: [] for r in entity_rela_list}  edi是每一个entity的编号
    def _load_entities(self, dataset):
        print('load entities...')
        num_nodes = 0
        data_relations, _, _ = dataset.get_relation()  # entity_relations, _, _
        entity_list = list(data_relations.keys())  #[USER,ITEM,FEATURE]
        for entity in entity_list:
            self.G[entity] = {}
            entity_size = getattr(dataset, entity).value_len    #entity的数量
            for eid in range(entity_size):
                entity_rela_list = data_relations[entity].keys()   #这个entity都有哪些relation
                self.G[entity][eid] = {r: [] for r in entity_rela_list}   #为entity类型实体 的 每一个eid 实体创建 对应的relation列表
            num_nodes += entity_size
            print('load entity:{:s}  : Total {:d} nodes.'.format(entity, entity_size))
        print('ALL total {:d} nodes.'.format(num_nodes))
        print('===============END==============')



    def _load_knowledge(self, dataset):
        _, data_relations_name, link_entity_type = dataset.get_relation()  # _, relation_name, link_entity_type

        for relation in data_relations_name:   #[INTERACT, FRIEND, LIKE, BELONG_TO]
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data    #对于这个relation  data = [[],...,[]]  是index node 有这个关系的 node 构成的list
            num_edges = 0
            for he_id, te_ids in enumerate(data):  # index是head_entity_id , te_ids是一个list,里面记录着tail_entity_ids
                if len(te_ids) <= 0:  #说明he_id没有这个关系的tail
                    continue
                e_head_type = link_entity_type[relation][0]   #relation 的 head 是什么类型的entity
                e_tail_type = link_entity_type[relation][1]   #relation 的 tail 是什么类型的entity
                for te_id in set(te_ids):
                    self._add_edge(e_head_type, he_id, relation, e_tail_type, te_id)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))
        print('===============END==============')

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data
