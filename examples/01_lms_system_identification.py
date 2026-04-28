#--------------------------------------------------------
#　本コードの概要
#　1. ホワイトノイズx, 真の未知FIR h_true, 希望信号 d = h_true * x を作成
#　2. LMSでh_trueの推定し、真値と比較
#　（2タップのみ抽出し放物面上の移動をアニメーションで観測）
#---------------------------------------------------------

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# examples から見て、1つ上がプロジェクト直下
ROOT = Path(__file__).resolve().parents[1]

# src フォルダを Python の探索先に追加
sys.path.append(str(ROOT / "src"))

from adaptive_audio.signals import make_white_noise, make_decay_fir, apply_fir
from adaptive_audio.lms import lms_identify


def main():
    # 実験設定
   
    fs = 16000
    n_samples = fs
    filter_len = 64
    mu = 0.01

    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 入力信号と真の未知FIRを作る
    x = make_white_noise(n_samples, seed=0)
    h_true = make_decay_fir(filter_len, seed=1)

    # x を h_true に通して、目標信号 d を作る
    d = apply_fir(x, h_true)

    # LMSで未知FIRを推定
    w_history, y, e = lms_identify(
        x=x,
        d=d,
        filter_len=filter_len,
        mu=mu,
    )

    h_est = w_history[-1]

    # 図1: 真のFIRと推定FIR
    plt.figure(figsize=(8, 4))
    plt.plot(h_true, label="true FIR")
    plt.plot(h_est, "--", label="estimated FIR")
    plt.xlabel("Tap")
    plt.ylabel("Amplitude")
    plt.title("LMS system identification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "01_lms_filter_estimation.png")
    plt.close()

    # 図2: 誤差パワーの時間変化
    plt.figure(figsize=(8, 4))
    plt.plot(10 * np.log10(e**2 + 1e-12))
    plt.xlabel("Sample")
    plt.ylabel("Error Power [dB]")
    plt.title("LMS error curve")
    plt.tight_layout()
    plt.savefig(figures_dir / "01_lms_error_curve.png")
    plt.close()

    print("Saved: 01_lms_filter_estimation.png")
    print("Saved: 01_lms_error_curve.png")
    








    
    # =========================
    # 動画: 2タップLMSの本物の3D放物面デモ
    # 保存はしない。実行時に表示だけする。
    #
    # 注意:
    # 上の64タップLMSとは別に、可視化専用として
    # filter_len = 2 の小さい問題をここで作る。
    # これにより、w[0], w[1], J の本物の放物面が描ける。
    # =========================

    demo_filter_len = 2
    demo_n_samples = 3000
    demo_mu = 0.03

    # 可視化用の入力信号
    x_demo = make_white_noise(demo_n_samples, seed=10)

    # 可視化用の真の2タップFIR
    h_demo_true = np.array([0.8, -0.4])

    # 目標信号 d_demo = h_demo_true * x_demo
    d_demo = apply_fir(x_demo, h_demo_true)

    # 2タップLMSを実行
    w_demo_history, y_demo, e_demo = lms_identify(
        x=x_demo,
        d=d_demo,
        filter_len=demo_filter_len,
        mu=demo_mu,
    )

    # 初期の0サンプル付近を除いて軌跡を間引く
    step = 5
    w0_path_full = w_demo_history[demo_filter_len - 1 :: step, 0]
    w1_path_full = w_demo_history[demo_filter_len - 1 :: step, 1]

    # 頂点付近に到達したらアニメーションを止める
    # tol を大きくすると早めに止まり、小さくすると頂点近くまで描く
    tol = 1e-3

    distance_to_true = np.sqrt(
        (w0_path_full - h_demo_true[0]) ** 2
        + (w1_path_full - h_demo_true[1]) ** 2
    )

    reached_indices = np.where(distance_to_true < tol)[0]

    if len(reached_indices) > 0:
        stop_frame = reached_indices[0] + 10  # 到達後、少しだけ余韻を残す
        stop_frame = min(stop_frame, len(w0_path_full))
    else:
        stop_frame = len(w0_path_full)

    w0_path = w0_path_full[:stop_frame]
    w1_path = w1_path_full[:stop_frame]

    # 表示範囲
    margin = 0.6
    w0_min = min(w0_path.min(), h_demo_true[0]) - margin
    w0_max = max(w0_path.max(), h_demo_true[0]) + margin
    w1_min = min(w1_path.min(), h_demo_true[1]) - margin
    w1_max = max(w1_path.max(), h_demo_true[1]) + margin

    w0_grid = np.linspace(w0_min, w0_max, 100)
    w1_grid = np.linspace(w1_min, w1_max, 100)
    W0, W1 = np.meshgrid(w0_grid, w1_grid)

    # コスト関数 J(w0, w1) = 平均二乗誤差
    # d_demo[n] - (w0*x[n] + w1*x[n-1]) を全サンプルで評価する
    eval_indices = np.arange(demo_filter_len - 1, demo_n_samples)

    J = np.zeros_like(W0)

    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w0 = W0[i, j]
            w1 = W1[i, j]

            err_sum = 0.0
            for n in eval_indices:
                x_vec = x_demo[n - demo_filter_len + 1 : n + 1][::-1]
                y_temp = w0 * x_vec[0] + w1 * x_vec[1]
                e_temp = d_demo[n] - y_temp
                err_sum += e_temp**2

            J[i, j] = err_sum / len(eval_indices)

    # LMS軌跡上のJも計算する
    z_path = []

    for w0, w1 in zip(w0_path, w1_path):
        err_sum = 0.0
        for n in eval_indices:
            x_vec = x_demo[n - demo_filter_len + 1 : n + 1][::-1]
            y_temp = w0 * x_vec[0] + w1 * x_vec[1]
            e_temp = d_demo[n] - y_temp
            err_sum += e_temp**2

        z_path.append(err_sum / len(eval_indices))

    z_path = np.array(z_path)

    # 真値の位置でのコスト
    true_err_sum = 0.0
    for n in eval_indices:
        x_vec = x_demo[n - demo_filter_len + 1 : n + 1][::-1]
        y_temp = h_demo_true[0] * x_vec[0] + h_demo_true[1] * x_vec[1]
        e_temp = d_demo[n] - y_temp
        true_err_sum += e_temp**2

    z_true = true_err_sum / len(eval_indices)

    # 3D図の作成
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        W0,
        W1,
        J,
        alpha=0.75,
        linewidth=0,
        antialiased=True,
    )

    # 真の係数、つまり放物面の頂点
    ax.scatter(
        h_demo_true[0],
        h_demo_true[1],
        z_true,
        marker="x",
        s=100,
        label="true coefficient",
    )

    # LMSの軌跡
    path_line, = ax.plot([], [], [], linewidth=2.0, label="LMS path")
    current_point, = ax.plot([], [], [], marker="o", markersize=6)

    ax.set_xlabel("w[0]")
    ax.set_ylabel("w[1]")
    ax.set_zlabel("Cost J")
    ax.set_title("LMS trajectory on true 2D quadratic cost surface")
    ax.legend()

    # 見やすい角度
    ax.view_init(elev=30, azim=-60)

    def update(frame):
        path_line.set_data(w0_path[: frame + 1], w1_path[: frame + 1])
        path_line.set_3d_properties(z_path[: frame + 1])

        current_point.set_data([w0_path[frame]], [w1_path[frame]])
        current_point.set_3d_properties([z_path[frame]])

        ax.set_title(
            f"LMS trajectory on true 2D quadratic cost surface | frame {frame}"
        )

        return path_line, current_point

    _ani = FuncAnimation(
        fig,
        update,
        frames=len(w0_path),
        interval=30,
        blit=False,
        repeat=False,
    )

    plt.show(block=True)




if __name__ == "__main__":
    main()