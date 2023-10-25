import pandas as pd
import json

def evaluator(pred_file, gt_file):

    correct_num = 0
    preded_num = 0
    ground_truth_num = 0

    Wrong_predictions = []
    Unpredictions = []

    preds = {row['LineID']: ['$$'.join([str(j) for j in i]) for i in json.loads(json.dumps(eval(row['pair'])))] for idx, row in pred_file.iterrows()}
    gts = {row['LineID']: ['$$'.join([str(j) for j in i]) for i in json.loads(json.dumps(eval(row['pair'])))] for idx, row in gt_file.iterrows()}
    assert len(preds) == len(gts), 'Prediction and groundtruth must have same length.'
    for lineID, pairs in preds.items():
        ground_truth_num += len(gts[lineID])
        preded_num += len(pairs)

        predicted = []
        for pair in pairs:
            if pair in gts[lineID]:
                correct_num += 1
                predicted.append(pair)
            else:
                Wrong_predictions.append({lineID: pair})
        unpredicted = [i for i in gts[lineID] if i not in predicted]
        if unpredicted:
            Unpredictions.append(unpredicted)

    recall = correct_num / ground_truth_num
    precision = correct_num / preded_num
    f1 = 2 * recall * precision / (recall + precision)

    # print(ground_truth_num)
    # print
    print('Precision: {}, Recall: {}, F1: {}'.format(precision, recall, f1))
    print('Wrong Prediction cases:', Wrong_predictions)
    print('Unprediction cases:', Unpredictions)

if __name__ == '__main__':

    systems = ['Andriod', 'Hadoop', 'HDFS', 'Linux', 'OpenStack', 'Spark', 'Zookeeper']
    for system in systems:
        print('Evaluating', system)
        pred_file = pd.read_csv('data/prediction/{}.csv'.format(system))
        gt_file = pd.read_csv('data/test/label/{}_gt.csv'.format(system))
        evaluator(pred_file, gt_file)
        print('---END---')

