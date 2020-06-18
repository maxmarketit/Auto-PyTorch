def qloss_metric(y_true, y_pred, qs):
        e = y_true - y_pred
        return np.maximum(qs * e, (qs-1)*e).mean()
## Usage
## metric_ql = functools.partial(qloss_metric, qs = np.array([[0.1, 0.5, 0.9]])) 
#              qs = quantiles for model output 1, 2, 3

def qinputloss_metric(y_true, y_pred, ws):
    w_q = ws[0,0]
    w_input = ws[0:1, 1:]
    q = y_true[:, 0:1]

    e_q = np.abs(y_true[:,0]-y_pred[:,0])
    e = y_true[:, 1:] - y_pred[:, 1:]

    ret = w_input * np.maximum(q*e, (q-1)*e)

    return np.mean(ret)+ np.mean(w_q*e_q)
## Usage
#  metric_qil = functools.partial(qinputloss_metric, ws = np.array([[0, 0.5, 0.5]]))
#               ws = weights for q and model output 1, 2  