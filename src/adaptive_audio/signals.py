#　nサンプルのホワイトノイズ生成
import numpy as np
def make_white_noise(n_samples: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples)

#　減衰するFIRフィルタの作成（真の未知FIRフィルタh_trueの作成）
def make_decay_fir(filter_len: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(filter_len)
    
    #音響経路に寄せた時間領域の指数減衰を与える
    decay = np.exp(-np.linspace(0.0, 4.0, filter_len))
    h = h * decay 
    h = h / np.linalg.norm(h)
    return h

#　xをh_trueに通してdを作る関数
def apply_fir(x: np.ndarray, h: np.ndarray):
    y = np.convolve(x, h)
    # 扱いやすくするためyとxの長さを揃える
    y = y[:len(x)]
    return y
