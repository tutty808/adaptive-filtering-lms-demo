import numpy as np

#　LMSによるFIRシステム同定（入力x,希望信号d,推定するフィルタ長d,ステップサイズmu) 
def lms_identify(x: np.ndarray, d: np.ndarray, filter_len: int, mu: float):
    n_samples = len(x)

    #　推定フィルタ係数（初期値0)
    w = np.zeros(filter_len)
    #　記録用
    w_history = np.zeros((n_samples, filter_len))
    y = np.zeros(n_samples)
    e = np.zeros(n_samples)

    #　LMS更新ループ
    for n in range(filter_len - 1, n_samples):

        #　x[n]からx[0]へfilter_len個(n+1個)並べる
        x_vec = x[n - filter_len + 1 : n+1][::-1]
        #　現在の推定出力
        y[n] = np.dot(w, x_vec)
        #　誤差
        e[n] = d[n] - y[n]
        #　LMS更新
        w = w + mu * e[n] * x_vec
        #　保存
        w_history[n] = w

    return w_history, y, e
    