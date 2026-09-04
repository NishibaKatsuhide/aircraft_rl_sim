# 3D Aircraft Obstacle-Avoidance RL Simulator

航空機が3次元空間を飛行し、地面から無限上空へ伸びる円柱状障害物を避けながら、指定されたゴール上空へ到達する強化学習用シミュレータです。

## 設計

- 環境: Gymnasium `Env`
- 強化学習: 最新の公式 Ray/RLlib API を想定
- アルゴリズム: PPO
- NN: RLlib の標準 MLP (`256-256`)
- Action: 絶対方位角 [-pi, pi]
- Observation:
  - 自機位置 `(x, y, z)`
  - ゴール位置 `(x, y)`
  - 最大12個の障害物 `(x, y, active)`
- 障害物: 地面から上方へ無限に伸びる円柱
- 航空機:
  - 水平速度一定
  - 指定方位へ最大旋回率まで追従
  - 高度は簡易オートパイロットで上昇・巡航・降下
- 成功:
  - ゴール中心から水平75 m以内
  - ゴール高度 ±50 m以内
- 衝突:
  - 障害物円柱の半径 + 航空機安全半径以内
- 報酬:
  - 毎step: -1
  - ゴール: `1000 - 2 * step_count`
  - 衝突: -1000
- 学習中:
  - `training_history.csv`
  - TensorBoard
  - 一定間隔で評価飛行の3D軌跡PNG
  - checkpoint
- 学習済みモデル:
  - `visualize.py` で3D軌跡をアニメーション表示

## インストール

Python 3.10～3.12程度を推奨。

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

RLlib公式ドキュメントでは `pip install -U "ray[rllib]"` が案内されています。
このプロジェクトでは依存関係を固定しすぎず、インストール時点の最新RLlibを使用します。

## 学習

```bash
python train.py --iterations 300
```

GPUを使う場合:

```bash
python train.py --iterations 1000 --num-learners 1 --gpus-per-learner 1
```

学習結果は `runs/` 以下に保存されます。

TensorBoard:

```bash
tensorboard --logdir runs
```

## 学習途中の飛行を見る

学習中に保存された checkpoint を指定:

```bash
python visualize.py --checkpoint runs/checkpoints/iter_0100
```

または最後のcheckpoint:

```bash
python visualize.py --checkpoint <checkpoint-directory>
```

`--episodes`、`--save` などの引数も利用できます。

## 重要なモデル化上の注意

今回の仕様では障害物が「地面から無限上空まで伸びる円柱」なので、高度を上げても障害物を回避できません。
したがって、障害物回避という観点では本質的には水平2次元の経路計画問題です。
3次元化は「自機位置を3次元で持つ」「ゴール上空への到着」「上昇・巡航・降下」の部分に意味があります。

もし将来、
- 高度制限付き障害物
- 山岳
- 飛行禁止空域
- 最低/最高飛行高度
- 速度制御
- 上昇/降下率制御
- バンク角・ロール・ピッチ
- 風
- GPS/INS誤差

などを入れるなら、Observation/Actionを拡張できます。
