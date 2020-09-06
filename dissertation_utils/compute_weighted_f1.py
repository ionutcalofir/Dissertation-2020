num_examples = [[6581, 12565, 1274], [6578, 12572, 1271]]
f1s = [[0.8998, 0.9445, 0.9200], [0.8892, 0.9454, 0.8544]]

for idx, (num_example, f1) in enumerate(zip(num_examples, f1s)):
    print('Weighted f1: {}'.format((num_example[0] * f1[0] + num_example[1] * f1[1] + num_example[2] * f1[2]) / sum(num_example)))
