import re
import numpy as np
from typing import List
from sklearn.tree import export_text
from sklearn.ensemble.forest import RandomForestClassifier

def print_tree( lines : List[str] = None, features : List[str] = None) -> List[str]:
    
    '''
    lines : DecisionTree export text
    
    '''
        
    condition_seq = [re.search('---',l).span()[0] for l in lines[:-1]]
    vba_script = []
    max_ = max(condition_seq) #leaf node


    #output only prediction , probability output will be added for the final training phase, now cant make sure to calculate right prob as tree does not provide



    
    
    set_ = np.unique(condition_seq)
    for i, l in enumerate(lines[:-1]):
        ### leaf 
        if condition_seq[i] == max_ :
            prediction = re.search('[0-9]',l).group()
            n_space = np.where(set_ == max_)[0][0]
            vba_script += [' '* n_space + f'prob = {prediction}']
            if 'Else' in vba_script[-2]:
                vba_script += [' '* (n_space-1) + 'End If']
                
            continue


        str_start_pos = re.search('[A-Za-z]',l).span()[0]
        cmd = l[str_start_pos:] #the logic

        ### top branch
        if condition_seq[i] == 1:
            if i == 0:
                head = 'If '
                vba_script += [head + cmd + ' Then']
            else:
                if 'Else' in vba_script[-3]:
                    vba_script += ['End If']
                head = 'Else: '
                vba_script += [head]

        else:
            n_space = np.where(set_ == condition_seq[i])[0][0]
            if condition_seq[i] > condition_seq[i-1]: #child
                head = ' '* n_space +'If '
                vba_script += [head + cmd + ' Then']
            else:
                head = ' '* n_space + 'Else: '
                vba_script += [head ]

    
    vba_script += ['End If']
    
    
    return vba_script


def print_vba(model, features):
    vba = []
    vba += ['Sub IcuRiskModel()']
    
    #declare var
    for f in features:
        vba += [f'  Dim {f} As Double'] #every feature can be float
    vba += [f'  Dim outcome({len(model.estimators_) -1}) As Double']
    vba += ['']
    
    #declare the input location for the dim
    excel_horiz = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # at most 26 features
    for alp, f in enumerate(features):
        vba += [f'{f} = Range(\"{excel_horiz[alp]}2\").Value'] #tentatively
    vba += ['']
    if isinstance(model, RandomForestClassifier):
        # Dim trees(len(model.estimators_)) As Double
        for i,m in enumerate(model.estimators_) :
            lines = export_text(m, feature_names=features).split('\n')
            vba += print_tree(lines, features)
            vba += ['']
            vba += [f'outcome({i}) = prob']
    vba += ['']
    vba += ['End If']
    vba += ['If Application.WorksheetFunction.Average(outcome) > 0.5 Then prob = 1 Else: prob = 0']

    vba += [f'Range(\"{excel_horiz[alp+1]}2\").Value = prob'] 
    vba += ['End Sub']
            
    return vba
