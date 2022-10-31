
import evaluation

dataset == 'f30k'

if dataset == 'f30k':
    evaluation.evalrank(model_path='',
                        data_path='',
                        split='test', save_path='')
else:
    evaluation.evalrank_eccv(model_path='',
                             data_path='',
                             split='test')
