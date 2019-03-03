# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Scratch code to convert SQuAD data into CoQA format

import json

data = {'version': 0.2, 'data': []}
with open('train-v2.0.json', 'r') as f:
    squad = json.load(f)

cnt = 0
for d in squad['data']:
    for p in d['paragraphs']:
        cnt += 1
        a = {'source': 'source', 'filename': 'filename', 'id': str(cnt), 'story': p['context']}
        ques = []
        ans = []

        additional = []
        turn_id = 1
        for qa in p['qas']:
            ques.append({
                    'input_text': qa['question'],
                    'turn_id': turn_id
                })

            if qa['is_impossible']:
                if len(ans) == 0:
                    ans.append([])
                ans[0].append({
                    'input_text': 'unknown',
                    'span_text': 'unknown',
                    'span_start': -1,
                    'span_end': -1,
                    'turn_id': turn_id
                })    

            for j in range(len(qa['answers'])):
                if j >= len(ans):
                    ans.append([])
                ans[j].append({
                    'input_text': qa['answers'][j]['text'],
                    'span_text': qa['answers'][j]['text'],
                    'span_start': qa['answers'][j]['answer_start'],
                    'span_end': qa['answers'][j]['answer_start'] + len(qa['answers'][j]['text']),
                    'turn_id': turn_id
                })    

            turn_id += 1

        a['questions'] = ques
        a['answers'] = ans[0]
        a['additional_answers'] = {}
        for j in range(1, len(ans)):
            a['additional_answers'][str(j-1)] = ans[j]

        data['data'].append(a)


with open('squad_in_coqa_format.json', 'w') as output_file:
    json.dump(data, output_file, indent=4)
