import argparse
import nltk.translate.gleu_score as gleu

def sentence_score(reference, hypothesis):
    '''
    input : reference - list of reference sentence
    hypothesis - list of hypothesis sentence
    '''
    tp, fp, fn = 0,0,0

    reference_set = set(reference)
    hypothesis_set = set(hypothesis)

    for token in hypothesis:
        if token in reference_set:
            tp += 1
        else:
            fp += 1
    
    for token in reference:
        if token not in hypothesis_set:
            fn += 1
    return tp, fp, fn

def corpus_macro_score(ref_line, hypo_line):
    tp, fp, fn = 0,0,0

    for ref, hypo in zip(ref_line, hypo_line):
        sam_tp, sam_fp, sam_fn = sentence_score(ref.split(), hypo.split())
        tp += sam_tp
        fp += sam_fp
        fn += sam_fn
    
    return tp, fp, fn

def precision(tp, fp):
    return float(tp) / (tp + fp) if (tp + fp) > 0 else 0.
    

def recall(tp, fn):
    if (tp+ fn) == 0:
        return 0.
    return float(tp) / (tp + fn)

def f_measure(tp, fp, fn, beta=1):
    f_percision = precision(tp, fp)
    f_recall = recall(tp, fn)
    if f_percision == 0 or f_recall == 0:
        return 0.
    else:
        return (1 + beta ** 2) *(f_percision * f_recall) / ((beta **2 *f_percision) + f_recall)
'''
def accuracy_macro_score(ref_line, hypo_line):
    correct = 0
    w = open('./different_result.txt','w')    
    for ref, hypo in zip(ref_line, hypo_line):
        if ref.strip() == hypo.strip():
            correct += 1
        else:
            w.write('---------------------\n'+ref.strip()+'\n'+hypo.strip()+'\n--------------------------\n')
    w.close()
    return correct, len(ref_line)    
'''
def accuracy_macro_score_final(ref_line, hypo_line):
    correct = 0
    total = 0
    w = open('./different_result.txt','w')
    for ref, hypo in zip(ref_line, hypo_line):
        ref = ref.strip()
        hypo = hypo.strip()
        flag = 0
        for index in range(len(ref)):
            try:
                if ref[index] != hypo[index]:
                    flag = 1
                else:
                    correct += 1
            except IndexError:
                flag = 1
                continue
        total += len(ref)
        if flag == 1:
            w.write('---------------------\n'+ref.strip()+'\n'+hypo.strip()+'\n--------------------------\n')
    w.close()
    return correct, total

def accuracy(length, correct):
    return float(correct) / length

def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--true', type=str, help='correct spelling file')
    parser.add_argument('--pred', type=str, help='error spelling file')
    args = parser.parse_args()

    t = open(args.true,'r', encoding='utf8')
    p = open(args.pred,'r', encoding='utf8')

    tline = t.readlines()
    pline = p.readlines()

    tp, fp, fn = corpus_macro_score(tline, pline)
    #total_score, length = accuracy_macro_score(tline, pline)
    #total, correct = accuracy_macro_score(tline, pline)
    correct, total = accuracy_macro_score_final(tline, pline)
    gleu_result = score_gleu(tline, pline)

    print('Precision : ',precision(tp, fp), 'Recall : ',recall(tp, fn), 'F1 : ',f_measure(tp,fp,fn) )
    print('F0.5 : ',f_measure(tp,fp,fn,beta=0.5))
    print('Acc : ', accuracy(total, correct))
    print('GLEU : ', gleu_result)
