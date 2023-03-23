
import sys

def MSE(preds, golds):
    curr_MSE = 0
    for i in range(len(preds)):
        curr_MSE += (float(preds[i])-float(golds[i]))**2
    curr_MSE = float(curr_MSE)/float(len(golds))
    return curr_MSE

if __name__ == '__main__':


    predictions_fname = sys.argv[1]
    gold_standard_fname = sys.argv[2]

    predictions = [x.strip() for x in open(predictions_fname,
                                           encoding='utf8')]

    golds = [x.strip() for x in open(gold_standard_fname,
                                     encoding='utf8')]

    assert len(predictions) == len(golds), "Different number of predictions and golds"
    print('MSE: ', str(MSE(predictions, golds)))



