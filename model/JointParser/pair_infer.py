from JointParser.knowledge import Knowledge
import pandas as pd
import copy
import json
import re

def load_info(df_file):
    output = list()
    for index, row in df_file.iterrows():
        tmp_output = dict()
        tmp_output['log'] = json.loads(row['log'])
        tmp_output['pairs'] = json.loads(row['pairs'])
        tmp_output['cand_concept_id'] = json.loads(row['concept'])
        tmp_output['cand_instance_id'] = json.loads(row['instance'])
        output.append(tmp_output)

    return output

def correct_pair(mentionA, mentionB, concept_ids, instance_ids):
    if mentionA in concept_ids and mentionB in instance_ids:
        return mentionA, mentionB
    elif mentionA in instance_ids and mentionB in concept_ids:
        return mentionB, mentionA
    else:
        return 'None', 'None'

def pair_match(logs):

    knowledge = Knowledge()

    # collect all knowledge pairs from one file.
    # process as dicts
    for log in logs:
        log['real_pair'] = []
        log['real_concept'] = []
        log['real_instance'] = []
        log['left_instance'] = []
        log['left_concept'] = []
        for pair in log['pairs']:
            concept_mention, instance_mention = correct_pair(pair[0], pair[1], log['cand_concept_id'], log['cand_instance_id'])

            if concept_mention != 'None':
                log['real_pair'].append([log['log'][concept_mention], log['log'][instance_mention]])
                knowledge.add_pair(log['log'][concept_mention], log['log'][instance_mention])

        log['left_instance'] = [log['log'][i] for i in log['cand_instance_id'] if
                                log['log'][i] not in [p[1] for p in log['real_pair']]]

        log['real_concept'] = [log['log'][i] for i in log['cand_concept_id']]

    ENDFLAG = 1
    match_num = 0
    while (ENDFLAG > 0):
        ENDFLAG = 0
        for log in logs:
            match_instance = []
            for idx, instance in enumerate(log['left_instance']):
                concepts = knowledge.find_concept_by_instance(instance)

                # if concepts:
                #     # print(instance, concepts)
                #     for concept in concepts:
                #         log['real_pair'].append([concept, instance])
                #         match_instance.append(idx)
                #     match_num += 1

            log['left_instance'] = [log['left_instance'][i] for i in range(len(log['left_instance'])) if
                                    i not in match_instance]
            if match_instance:
                ENDFLAG = 1


    for log in logs:
        log['left_concept'] = [i for i in log['real_concept'] if i not in [p[0] for p in log['real_pair']]]
        log['left_concept'] = list(set(log['left_concept']))
        log['left_instance'] = list(set(log['left_instance']))


    return logs


def conceptualize(logs):

    for log in logs:
        # replace pair
        log['conceptualized'] = copy.deepcopy(log['log'])
        log['params'] = []

        instance2concept = {i[1]:i[0] for i in log['real_pair']}
        # print(instance2concept)
        for idx, token in enumerate(log['log']):
            if token in instance2concept.keys():
                log['conceptualized'][idx] = '<*{}*>'.format(instance2concept[token])
                log['params'].append(token)

            if token in log['left_instance']:
                log['conceptualized'][idx] = '<*>'
                log['params'].append(token)

    return logs

def joint_infer(logs, pairs, concepts, instances):


    all_logs = []
    for log, pair, concept, instance in zip(logs, pairs, concepts, instances):
        tmp_log = {}
        tmp_log['log'] = log
        tmp_log['pairs'] = pair
        tmp_log['cand_concept_id'] = concept
        tmp_log['cand_instance_id'] = instance
        all_logs.append(tmp_log)

    outputs = pair_match(all_logs)
    outputs = conceptualize(outputs)
    # print(outputs)

    real_pair = [item['real_pair'] for item in outputs]
    left_concept = [item['left_concept'] for item in outputs]
    left_instance = [item['left_instance'] for item in outputs]
    conceptualized = [item['conceptualized'] for item in outputs]
    params = [item['params'] for item in outputs]

    return real_pair, left_concept, left_instance, conceptualized, params


