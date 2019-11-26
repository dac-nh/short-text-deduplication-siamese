from flair.data import Senence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbedding, ELMoEmbeddings
import tqdm.tqdm_notebook as tqdm


def stacking_embedding(df):
    glove_embedding = WordEmbeddings('glove')
    flair_embedding_news_forward = FlairEmbeddings('news-forward')
    flair_embedding_news_backward = FlairEmbeddings('news-backward')
    bert_embedding = BertEmbeddings()
    elmo_embedding = ELMoEmbeddings()


    stacked_embeddings = StackedEmbeddings([
        glove_embedding, 
        flair_embedding_news_forward, 
        flair_embedding_news_backward,
        bert_embedding,
        elmo_embedding
    ])
    
    for index, row tqdm(df.iterrows(), total=len(df), desc='Embedding'):
        sentence = Sentence(row['name'])
        token_series = set()
        for token in sentence:
            token_series.add