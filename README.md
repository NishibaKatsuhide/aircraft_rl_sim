# 3D Aircraft Obstacle-Avoidance RL Simulator

航空機が3次元空間を飛行し、地面から無限上空へ伸びる円柱状障害物を避けながら、指定されたゴール上空へ到達する強化学習用シミュレータです。

## 設計

- 環境: Gymnasium `Env`
- 強化学習: Ray/RLlib の PPO
- アルゴリズム: PPO
- NN: RLlib の標準 MLP (`256-256`)
- Action: デフォルトで離散（12-bin, 30°刻み）。連続角度（[-pi, pi]）も `discrete_actions` 設定で選べます。
- Observation:
  - 自機位置 `(x, y, z)`
  - ゴール位置 `(x, y)`
  - 最大12個の障害物 `(x, y, active)`
- 地理条件:
  - 空域は `x, y = 0..5000 [m]`
  - ゴールも同じ範囲にランダム配置
  - 航空機は初期状態では常にこの範囲内に出現
- 障害物:
  - 初期状態では 0 個
  - ゴール到達ごとに 1 個ずつ増加
  - 最大 10 個まで（`max_obstacles` で変更可能）
  - 地面から上方へ無限に伸びる円柱
- ゴール判定:
  - 高度は無視
  - 水平距離だけで判定
  - 閾値は 1000 → 800 → 600 → 400 → 200 m と段階的に狭くなる
  - 200 m で達成した後に障害物数が 1 増える
- 境界:
  - `x` または `y` が 0 または 5000 を超えると境界衝突
  - 罰則は `-500` で episode 終了
- 航空機:
  - 水平速度一定
  - 指定方位へ最大旋回率まで追従
  - 高度は簡易オートパイロットで上昇・巡航・降下
- 衝突:
  - 障害物円柱の半径 + 航空機安全半径以内
- 報酬:
  - 毎step: -1
  - ゴール: `1000 - 2 * step_count`
  - 障害物衝突: -1000
  - 境界衝突: -500
- 学習中:
  - `training_history.csv`
  - TensorBoard
  - 一定間隔で評価飛行の3D軌跡PNG
  - `runs/aircraft_ppo/state_logs/*.json` に episode 状態を保存（`visualize.py` はこれを再生します）
  - checkpoint
  - 学習安定化のための対策: 観測値の非有限値除去、報酬/行動のクリップ、`log_std` のクリップ、障害物増加時の学習率半減などが組み込まれています
- 学習済みモデル:
  - `visualize.py` で3D軌跡をアニメーション表示
  - 可能なら `--state-log` で保存済み episode を再生

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

注意: デフォルト設定では障害物の最大数が `10`、行動は離散（12-bin）となっています。

TensorBoard:

```bash
tensorboard --logdir runs
```

## 学習途中の飛行を見る

保存済み episode のログを再生する（推奨）:

```bash
python visualize.py --state-log runs/aircraft_ppo/state_logs/iter_00041.json
```

保存済み checkpoint から直接再現する場合:

```bash
python visualize.py --checkpoint runs/aircraft_ppo/checkpoints/iter_00300
```

保存先を指定して GIF などに出力する場合:

```bash
python visualize.py --state-log runs/aircraft_ppo/state_logs/iter_00041.json --save /tmp/episode.gif
```

利用できる主な引数は `--checkpoint` / `--state-log` / `--seed` / `--save` です。

## 重要なモデル化上の注意

今回の仕様では障害物が「地面から無限上空まで伸びる円柱」なので、高度を上げても障害物を回避できません。
したがって、障害物回避という観点では本質的には水平2次元の経路計画問題です。
3次元化は「自機位置を3次元で持つ」「ゴール上空への到着」「上昇・巡航・降下」の部分に意味があります。

また、現在のゴール判定は高度を無視し、水平距離の閾値だけで評価します。
そのため、ゴールの視覚的な半径は `1000, 800, 600, 400, 200 [m]` と変化し、障害物数も同時に進行します。

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
