import json
from pathlib import Path
import spacy
import scispacy
import pandas as pd


# Use the large biomedical scispacy model since we want to find named entities.
nlp = spacy.load('en_core_sci_lg')


FILES = '/env/covid-19/data'


def make_df(file):
    """Read text fields from JSON file into spacy docs; create dataframe with
    entities, title (so we can later filter by topic) and source file (so we
    can locate the source data)."""
    p = Path(file)
    with p.open() as f:
        data = json.load(f)
        texts = []
        try:
            abstract = data['abstract'][0]['text']
            texts.append(abstract)
        except (IndexError, KeyError):
            print('No abstract.')
        for i in range(len(data['body_text'])):
            text = data['body_text'][i]['text']
            texts.append(text)
        # for efficiency, use nlp.pipe on small chunks of texts and NER only
        for doc in nlp.pipe(texts, disable=["tagger", "parser"]):
            try:
                ents = [[ent.text, p.stem] for ent in doc.ents]
                return pd.DataFrame(ents, columns=['Entity', 'Source'])
            except AttributeError:
                print('No entities found.')


def get_context(file, keyword):
    """Read text fields from JSON file into spacy docs; return list of
    sentences that include the target word."""
    Path(FILES)
    q = sorted(Path().rglob(file))
    for item in q:
        with item.open() as f:
            data = json.load(f)
            texts = []
            try:
                abstract = data['abstract'][0]['text']
                texts.append(abstract)
            except (IndexError, KeyError):
                print(file)
            for i in range(len(data['body_text'])):
                text = data['body_text'][i]['text']
                texts.append(text)
            sents = []
            for doc in nlp.pipe(texts):
                for sent in doc.sents:
                    for token in sent:
                        if token.text == keyword:
                            if sent not in sents:
                                sents.append(sent)
        return sents


# Create dataframe listing all entities in all full-text articles (3 hr run)
p = Path(FILES)
df = pd.concat(make_df(file) for file in p.glob('**/*.json'))


# View entities; write csv with all items
entities = df['Entity'].value_counts().to_dict()
sorted_entities = sorted(entities.items())

with open('entities.csv', 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerows(sorted_entities)


# Load metadata file as dataframe
metadata = '/env/covid-19/data/metadata.csv'
sources = pd.read_csv(metadata)


# find filename (using sha from initial dataframe) for each keyword
def make_final_table(entity, keyword):
    entity_filter = df[df['Entity'].str.contains(entity)]  # entities
    entity_source = entity_filter['Source'].drop_duplicates().values.tolist()
    out = []
    for sha in entity_source:
        file = f'{sha}.json'
        snippets = str(get_context(file, keyword))  # sentence can contain a different keyword
        info = sources[sources['sha'].str.contains(sha, na=False)]
        if not info['publish_time'].empty:
            date = info['publish_time'].item()
        if not info['title'].empty:
            title = info['title'].item()
        if not info['url'].empty:
            url = info['url'].item()
            out.append([date, title, url, snippets])
    return out

def make_results_df(data):
    """Create the dataframe for each piece of relevant content, listing the
    date, title, and URL of the study, with list of matching sentences."""
    return pd.DataFrame(data, columns=['Date', 'Study', 'URL', 'Snippet'])


# RUN
keywords = ["Asymptomatic", "asymptomatic", "asymptomatic carrier phase", "Asymptomatic controls", "asymptomatic controls", "Asymptomatic infection", "asymptomatic infection", "Asymptomatic infections", "asymptomatic infections", "asymptomatic line", "asymptomatic-to-mild ones", "asymptomatic/mild infections", "asymptomatically", "clinically asymptomatic", "delay-adjusted asymptomatic", "prodromal/asymptomatic stages", "respiratory asymptomatic", "SARS-CoV-2 asymptomatic"]


# For each named entity, get the relevant background data
content = [make_final_table(entity, keyword='asymptomatic') for entity in entities]

# Concatenate dataframe with all results
results = pd.concat(make_results_df(data) for data in content)

# Sort with newest first
date_sort = results.sort_values(by='Date', ascending=False)

# Write csv
results.to_csv('asymptomatic.csv', index=False)
