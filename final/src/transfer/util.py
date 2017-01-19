import sys
def score(predictions, trues):
    total = [ 0, 0 ] # total weight/ total score
    for key in trues:
        weight = len(trues[key])
        total[0] += weight
        if key in predictions:
            add = weight * f1(predictions[key], trues[key])
            total[1] += weight * f1(predictions[key], trues[key])
        else:
            total[1] += 0
    print total
    return float(total[1] / total[0])

def f1(prediction, true):
    true_prediction = []
    wrong_prediction = []
    tp = 0
    fp = 0
    fn = 0
    p = 0
    r = 0
    for pred in prediction:
        if pred in true:
            true_prediction.append(pred)
        else:
            wrong_prediction.append(pred)
    tp = len(true_prediction)
    fp = len(wrong_prediction)
    fn = len(true) - tp
    if tp + fp != 0:
        p = float(tp) / float(tp + fp)
    if tp + fn != 0:
        r = float(tp) / float(tp + fn)
    if p + r != 0:
        score = float(2 * p * r) / float(p + r)
        return score
    else:
        return 0
