import string
import json

def clean(word):
    for w in word:
        if w not in string.punctuation:
            return True
    return False

def write(log, predicts, predict_token_label):
    # write for graph construction

    cur_output = {}
    cur_output['log'] = log
    cur_output['pairs'] = []
    cur_output['concept'] = []
    cur_output['instance'] = []

    for predict in predicts:
        if predict[1] != 0:
            mentionA_idx = predict[0]
            mentionB_idx = predict[1]
            if clean(log[mentionA_idx]) and clean(log[mentionB_idx]):
                cur_output['pairs'].append([mentionA_idx, mentionB_idx])

    predict_token_label = predict_token_label.tolist()
    for idx, label in enumerate(predict_token_label):
        if label == 1:
            cur_output['concept'].append(idx)
        elif label == 2:
            cur_output['instance'].append(idx)

    return cur_output
