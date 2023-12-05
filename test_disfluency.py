import nltk
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite.metrics import flat_f1_score
import joblib

def read_conll_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            line = line.strip()
            print(line)
            if line:
                word, pos_tag = line.split('\t')
                sentence.append((word, pos_tag))
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
        if sentence:
            sentences.append(sentence)
    return sentences

# Define features for CRF model
def word2features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    # first word
    if i==0:
        prevword = '<START>'
        prevpos = '<START>'
    else:
        prevword = sent[i-1][0]
        prevpos = sent[i-1][1]
        
    # first word
    if i==0 or i==1:
        prev2word = '<START>'
        prev2pos = '<START>'
    else:
        prev2word = sent[i-2][0]
        prev2pos = sent[i-2][1]
    
    # last word
    if i == len(sent)-1:
        nextword = '<END>'
        nextpos = '<END>'
    else:
        nextword = sent[i+1][0]
        nextpos = sent[i+1][1]
    
    # suffixes and prefixes
    pref_1, pref_2, pref_3, pref_4 = word[:1], word[:2], word[:3], word[:4]
    suff_1, suff_2, suff_3, suff_4 = word[-1:], word[-2:], word[-3:], word[-4:]
    
#    rule_state = rule_based_tagger.tag([word])[0][1]
    
    return {'word': word,
            # 'pref_1': pref_1,
            # 'pref_2': pref_2,
            # 'pref_3': pref_3,
            # 'pref_4': pref_4,
            # 'suff_1': suff_1,
            # 'suff_2': suff_2,
            # 'suff_3': suff_3,
            # 'suff_4': suff_4,
            'pos':pos,            
            'prevword': prevword,
            'prevpos': prevpos,
            'nextword': nextword,
            'nextpos': nextpos,
            'prev2word': prev2word,
            'prev2pos': prev2pos
           } 

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [disfluence for word, postag, disfluence in sent]

def sent2tokens(sent):
    return [word for word, postag in sent] 



# Path to your CoNLL format file
file_path = '/root/home_cdac/ICON/data/Test-Blind/Marathi/marathi_test_blind_postagged.tsv'

# Read tagged sentences from the CoNLL format file
tagged_sentences = read_conll_file(file_path)
#print(tagged_sentences[0])
# Prepare data for training
X = [sent2features(sent) for sent in tagged_sentences]

# Create CRF model
model_file_path = '/root/home_cdac/ICON/data/disfluency/Marathi/crf.model'
crf = loaded_model = joblib.load(model_file_path)

y_pred = crf.predict(X)
labels = list(crf.classes_)


output_file_path = '/root/home_cdac/ICON/data/disfluency/Marathi/CDACN_Submission_1.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    
    for i, sent_tags in enumerate(y_pred):
       # print(sent_tags)
        for j, tag in enumerate(sent_tags):
            word = X[i][j]['word']
            pos= X[i][j]['pos']
            output_file.write(f"{word}\t{tag}\n")
        output_file.write('\n')
