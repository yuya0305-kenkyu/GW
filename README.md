# 概要
重力波の到来方向推定に使うプログラムが保存されています。  
コードを使用するには `generate_hogehoge_data.py` を使ってデータの作成を行ってください。  
感度のデータ `sensitivities` は (https://dcc.ligo.org/LIGO-T2000012/public) から取得しています。  

MLPについては無編集、TCNについてはデータ拡張、チャンネル数の増加、カーネルサイズの増加など一部改変を行っています。

# TCNについて
・Global Max Poolingの実装  
・データリークの解消  
・データ拡張の追加  
・Early Stoppingの実装  
・チャンネル数の増加、カーネルサイズの増加  
・最適化アルゴリズムの更新(`adam`→`adamW`)と`weight_decay`の実装  
・`history`をcsv形式で書き出し  
