from utils import load_embed, load_timetrain_embed, load_timetrain_embed_processData, MOVIE, YELP_STAR
from utils import savePKL
import torch

def ndarray2tensor():
    embeds = load_embed('YELP_STAR', 'transe', epoch=0)
    new_embeds = {}
    new_embeds['ui_emb'] = torch.from_numpy(embeds['ui_emb'])
    new_embeds['feature_emb'] = torch.from_numpy(embeds['feature_emb'])
    savePKL(file='/data/user/zjh/recommend/code/TAF4CRS/tmp/yelp_star/embeds/transe-ndarray.pkl', obj=embeds)
    savePKL(file='/data/user/zjh/recommend/code/TAF4CRS/tmp/yelp_star/embeds/transe.pkl', obj=new_embeds)

TMP_DIR = {
    YELP_STAR: './tmp/yelp_star',
    MOVIE:'./tmp/movie'
}

def tensor2ndarray():
    # dataset,time_emb_file = 'MOVIE',1489
    dataset,time_emb_file = 'YELP_STAR', 1668
    embeds = load_timetrain_embed_processData(dataset,time_emb_file)
    embeds['ui_emb'] = embeds['ui_emb'].numpy()
    embeds['feature_emb'] = embeds['feature_emb'].numpy()
    embeds['time_emb'] = embeds['time_emb'].numpy()
    embeds['time_to_hidden'] = embeds['time_to_hidden'].numpy()
    embeds['user_time_transfer'] = embeds['user_time_transfer'].numpy()
    savePKL(file=TMP_DIR[dataset] + f'/Time-model-embeds/new-iter-{time_emb_file}.pkl', obj=embeds)

if __name__ == '__main__':
    tensor2ndarray()
    print('...')