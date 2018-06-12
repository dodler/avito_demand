import re
from pymystem3 import Mystem

# Таблица преобразования частеречных тэгов Mystem в тэги UPoS:
import requests

mapping_url = \
    'https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map'

mystem2upos = {}
r = requests.get(mapping_url, stream=True)
for pair in r.text.split('\n'):
    pair = pair.split()
    if len(pair) > 1:
        mystem2upos[pair[0]] = pair[1]

print('Loading the model...', file=sys.stderr)
m = Mystem()


def tag_mystem(text='Текст нужно передать функции в виде строки!', mapping=None, postags=True):

    # если частеречные тэги не нужны (например, их нет в модели), выставьте postags=False
    # в этом случае на выход будут поданы только леммы
    text = re.sub(r'[^\w\s]|_',' ',str(text))
    processed = m.analyze(text)
    tagged = []
    for w in processed:
        try:
            lemma = w["analysis"][0]["lex"].lower().strip()
            pos = w["analysis"][0]["gr"].split(',')[0]
            pos = pos.split('=')[0].strip()
            if mapping:
                if pos in mapping:
                    pos = mapping[pos]  # здесь мы конвертируем тэги
                else:
                    pos = 'X'  # на случай, если попадется тэг, которого нет в маппинге
            tagged.append(lemma.lower() + '_' + pos)
        except BaseException:
            continue  # я здесь пропускаю знаки препинания, но вы можете поступить по-другому
    if not postags:
        tagged = [t.split('_')[0] for t in tagged]
    return tagged


def get_idx4tags(tags):
    res = []
    for t in tags:
        if t in words2idx.keys():
            res.append(words2idx[t])

    return res