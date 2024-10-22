train.pyで訓練

eval4_image.pyで画像出力

eval5_pnsr_100.pyで評価指標と画像出力

conda activate cwvae
の中に入る

cwvaeを構成する重要なファイルは
cells.py→cwvaeの根幹部分で、RNNのような働きをするRSSMセルの定義をしています、
cnn.py→encoderとdecoderの層について定義してます。
cwvae.py→モデルのインスタンス化をしている部分です。
data_loader.py→データ取り出す部分です。
tool.py→色んな関数の定義しているコードです。
train.py→訓練させるコードです。


configsにbase_config.ymlとminerl.ymlがあり、重要なパラメータを定義しています。

設定ファイル
configs/base_config.yml: モデルの基本的な設定（例: 隠れ層のサイズ、学習率など）を記述。
configs/minerl.yml: データセット固有の設定（例: データパス、シーケンス長など）を記述。

