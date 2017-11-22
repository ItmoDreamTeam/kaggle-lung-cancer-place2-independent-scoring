from sklearn.metrics import log_loss


def nothing_transform(prob):
    return prob


def transform(prob):
    if prob >= 0.5:
        return 1
    else:
        return 0


# id -> probability
def read_file(file_name, transformer=nothing_transform):
    dic = {}
    with open(file_name) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            id, prob = line.split(',')
            prob = float(prob)
            dic[id] = transformer(prob)

    return dic


if __name__ == '__main__':
    predict = read_file(r"../scoring_code/final_predictions_dh_blend.csv", nothing_transform)
    expected = read_file(r"../scoring_code/julian_preds_test.csv", transform)

    pred_arr = []
    exp_arr = []

    for key in predict.keys():
        pred_arr.append(predict[key])
        exp_arr.append(expected[key])

    print 'Log Loss: {}'.format(log_loss(exp_arr, pred_arr))
