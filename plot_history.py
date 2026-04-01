import pandas as pd
import matplotlib.pyplot as plt
import argparse

# コマンドラインからCSVのパスを受け取れるようにする
parser = argparse.ArgumentParser()
parser.add_argument("csv_path", help="Path to the history CSV file")
args = parser.parse_args()

# CSVの読み込み
df = pd.read_csv(args.csv_path)

# グラフの描画設定 (横長に2つ並べる)
plt.figure(figsize=(12, 5))

# 1つ目：Lossのグラフ
plt.subplot(1, 2, 1)
plt.plot(df['train_loss'], label='Train Loss', color='blue')
plt.plot(df['val_loss'], label='Val Loss', color='orange')
plt.title('Learning Curve (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2つ目：Accuracyのグラフ
plt.subplot(1, 2, 2)
plt.plot(df['train_acc'], label='Train Acc', color='blue')
plt.plot(df['val_acc'], label='Val Acc', color='orange')
plt.title('Learning Curve (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# レイアウトを整えて表示
plt.tight_layout()
plt.show()