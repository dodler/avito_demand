import os
import os.path as osp
import pandas as pd
import numpy as np

dirs = [k for k in os.listdir() if k.startswith('blendme_')]
solutions = []
for d in dirs:
    path = osp.join(d, 'solution.csv')
    if osp.exists(path):
        solutions.append(pd.read_csv(path))

print('solutions loaded', len(solutions))


def blend_solutions(solution_list):
    res = pd.DataFrame()
    res['item_id'] = solution_list[0]['item_id']
    res['deal_probability'] = solution_list[0]['deal_probability']
    for i, sol in enumerate(solution_list[1:]):
        res['deal_probability'] = np.mean((res['deal_probability'], sol['deal_probability']), axis=0)

    return res


blend = blend_solutions(solutions)
print('blending done')
blend.to_csv('output.csv', index=False, columns=['item_id', 'deal_probability'])
