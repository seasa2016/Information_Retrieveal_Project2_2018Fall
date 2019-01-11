from time import time
import pandas as pd
import numpy as np

# File paths
file_dir = '../data/'
train_j = []
with open(file_dir+'train.json') as f:
    for line in f:
        train_j.append(eval(line))    # train_j (dict): 'edges', 'nodes', 'tokens'
test_j = []
with open(file_dir+'test.json') as f:
    for line in f:
        test_j.append(eval(line))
        
# Json to dataframe
def json_to_df(train_j):
    word_1 = []
    node_1 = []
    word_2 = []
    node_2 = []
    rel = []
    for i in range(len(train_j)):
        node_entity = dict()
        for j in range(len(train_j[i]['nodes'])):
            node_entity[ str(train_j[i]['nodes'][j][0]) ] = list(train_j[i]['nodes'][j][1].keys())[0]
        
        for j in range(len(train_j[i]['edges'])):
            tmp_1 = ''
            token_ix = train_j[i]['edges'][j][0][0]
            while(token_ix < train_j[i]['edges'][j][0][1]):
                #tmp_1.append(train_j[i]['tokens'][token_ix])
                tmp_1 += (str(train_j[i]['tokens'][token_ix]) + ' ')
                token_ix += 1
            tmp_2 = ''
            token_ix = train_j[i]['edges'][j][1][0]
            while(token_ix < train_j[i]['edges'][j][1][1]):
                #tmp_2.append(train_j[i]['tokens'][token_ix])
                tmp_2 += (str(train_j[i]['tokens'][token_ix]) + ' ')
                token_ix += 1
            
            word_1.append(tmp_1)
            word_2.append(tmp_2)
            node_1.append( node_entity[ str(train_j[i]['edges'][j][0]) ] )
            node_2.append( node_entity[ str(train_j[i]['edges'][j][1]) ] )
            rel.append(list(train_j[i]['edges'][j][2].keys())[0])
    
    train_df = pd.DataFrame()
    train_df['word_1'] = word_1
    train_df['node_1'] = node_1
    train_df['word_2'] = word_2
    train_df['node_2'] = node_2
    train_df['rel'] = rel
    
    return train_df

train_df = json_to_df(train_j)
test_df = json_to_df(test_j)


# Prepare features
import string

def add_feature(df):
    num_char_1 = []
    num_char_2 = []
    num_word_1 = []
    num_word_2 = []
    num_cap_1 = []
    num_cap_2 = []
    num_symb_1 = []
    num_symb_2 = []
    num_numb_1 = []
    num_numb_2 = []
    
    for i in range(len(df)):
        num_char_1.append(len(df['word_1'][i]))
        num_char_2.append(len(df['word_2'][i]))
        num_word_1.append( df['word_1'][i].count(' '))
        num_word_2.append( df['word_2'][i].count(' '))
        num_cap_1.append(sum(1 for c in df['word_1'][i] if c.isupper()) / df['word_1'][i].count(' '))
        num_cap_2.append(sum(1 for c in df['word_2'][i] if c.isupper()) / df['word_2'][i].count(' '))
        num_symb_1.append(sum(not c.isalnum() for c in df['word_1'][i]) - df['word_1'][i].count(' '))
        num_symb_2.append(sum(not c.isalnum() for c in df['word_2'][i]) - df['word_2'][i].count(' '))
        num_numb_1.append(sum(c.isnumeric() for c in df['word_1'][i]))
        num_numb_2.append(sum(c.isnumeric() for c in df['word_2'][i]))
    
    df['num_char_1'] = num_char_1
    df['num_char_2'] = num_char_2
    df['num_word_1'] = num_word_1
    df['num_word_2'] = num_word_2
    df['num_cap_1'] = num_cap_1
    df['num_cap_2'] = num_cap_2
    df['num_symb_1'] = num_symb_1
    df['num_symb_2'] = num_symb_2
    
    return df

train_data = add_feature(train_df)
test_data = add_feature(test_df)

# Attributes
col_drop = ['word_1', 'word_2', 'rel']
train_data = train_data.drop(columns=col_drop)
test_data = test_data.drop(columns=col_drop)

attr_col = train_data.columns.values[0:].tolist()
X_train = train_data[attr_col].as_matrix()
X_test = test_data[attr_col].as_matrix()


# Encode relation types to labels
rel_type = list(set(train_df['rel']))
rel2label = dict()
label2rel = dict()
for i in range(len(rel_type)):
    rel2label[rel_type[i]] = i
    label2rel[i] = rel_type[i]

def encode_relation(df):
    rel_list = []
    for i in range(len(df)):
        rel_list.append( rel2label[df['rel'][i]] )

    return rel_list

y_train = encode_relation(train_df)
y_test = encode_relation(test_df)

## Classification
def multi_classification():
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score
    
    categorical_features_indices = [0,1]

    clf = CatBoostClassifier(
        loss_function='MultiClass',
        iterations=1000,
        random_seed=42,
        logging_level='Silent'     
    )

    clf.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        #eval_set=(X_validation, y_validation),
        #logging_level='Verbose',  
        #plot=True
    )
    # Prediction
    y_pred = clf.predict(X_test)
    
    from sklearn.metrics import classification_report, accuracy_score
    print ('\n classification report:\n', classification_report(y_test, y_pred))
    
    from sklearn.metrics import f1_score
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print ('f1-score (micro): ', f1_micro)
    print ('f1-score (macro): ', f1_macro)
    
    # Feature Importance
    feature_importances = clf.get_feature_importance(X=X_train,y=y_train, cat_features=categorical_features_indices)
    feature_names = train_data.columns
    print ('\n Feature Importance: ')
    for score, name in zip(feature_importances, feature_names):
        print('{}: {}'.format(name, score))
    
    #return f1_macro
multi_classification()
