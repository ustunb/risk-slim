import numpy as np

# ACCURACY STATISTICS COMPUTATION
def get_prediction(x, rho):
    return np.sign(x.dot(rho))


def get_true_positives_from_pred(yhat, pos_ind):
    return np.sum(yhat[pos_ind] == 1)


def get_false_positives_from_pred(yhat, pos_ind):
    return np.sum(yhat[~pos_ind] == 1)


def get_true_negatives_from_pred(yhat, pos_ind):
    return np.sum(yhat[~pos_ind] != 1)


def get_false_negatives_from_pred(yhat, pos_ind):
    return np.sum(yhat[pos_ind] != 1)


def get_accuracy_stats(model, data, error_checking=True):
    accuracy_stats = {
        'train_true_positives': np.nan,
        'train_true_negatives': np.nan,
        'train_false_positives': np.nan,
        'train_false_negatives': np.nan,
        'valid_true_positives': np.nan,
        'valid_true_negatives': np.nan,
        'valid_false_positives': np.nan,
        'valid_false_negatives': np.nan,
        'test_true_positives': np.nan,
        'test_true_negatives': np.nan,
        'test_false_positives': np.nan,
        'test_false_negatives': np.nan,
    }

    model = np.array(model).reshape(data['X'].shape[1], 1)

    # training set
    data_prefix = 'train'
    X_field_name = 'X'
    Y_field_name = 'Y'
    Yhat = get_prediction(data['X'], model)
    pos_ind = data[Y_field_name] == 1

    accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

    if error_checking:
        N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                   accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                   accuracy_stats[data_prefix + '_' + 'false_positives'] +
                   accuracy_stats[data_prefix + '_' + 'false_negatives'])
        assert data[X_field_name].shape[0] == N_check

    # validation set
    data_prefix = 'valid'
    X_field_name = 'X' + '_' + data_prefix
    Y_field_name = 'Y' + '_' + data_prefix
    has_validation_set = (X_field_name in data and
                          Y_field_name in data and
                          data[X_field_name].shape[0] > 0 and
                          data[Y_field_name].shape[0] > 0)

    if has_validation_set:

        Yhat = get_prediction(data[X_field_name], model)
        pos_ind = data[Y_field_name] == 1
        accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

        if error_checking:
            N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                       accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                       accuracy_stats[data_prefix + '_' + 'false_positives'] +
                       accuracy_stats[data_prefix + '_' + 'false_negatives'])
            assert data[X_field_name].shape[0] == N_check

    # test set
    data_prefix = 'test'
    X_field_name = 'X' + '_' + data_prefix
    Y_field_name = 'Y' + '_' + data_prefix
    has_test_set = (X_field_name in data and
                    Y_field_name in data and
                    data[X_field_name].shape[0] > 0 and
                    data[Y_field_name].shape[0] > 0)

    if has_test_set:

        Yhat = get_prediction(data[X_field_name], model)
        pos_ind = data[Y_field_name] == 1
        accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

        if error_checking:
            N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                       accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                       accuracy_stats[data_prefix + '_' + 'false_positives'] +
                       accuracy_stats[data_prefix + '_' + 'false_negatives'])
            assert data[X_field_name].shape[0] == N_check

    return accuracy_stats


# ROC Curve + AUC
def get_roc_metrics(model, data):
    pass

def get_calibration_metrics(model, data):
    #distinct scores
    #calibration error at each score
    #max calibration error
    #mean calibration error
    scores = (data['X'] * data['Y']).dot(model)
    pass


