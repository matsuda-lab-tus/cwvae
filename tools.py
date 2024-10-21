import yaml  # YAML形式の設定ファイルを読み込むためのライブラリ
from pathlib import Path  # パス操作を便利にするライブラリ
import torch  # PyTorchライブラリをインポート
from torch import nn  # PyTorchのニューラルネットワークのモジュールをインポート
import numpy as np  # 数値計算を行うためのライブラリ
import imageio  # 画像を読み込むためのライブラリ
import os  # オペレーティングシステムに関する機能を提供するライブラリ
import matplotlib.pyplot as plt  # グラフを描くためのライブラリ
from skimage.metrics import structural_similarity as ssim_metric  # SSIM（構造類似度）を計算するための関数
from skimage.metrics import peak_signal_noise_ratio as psnr_metric  # PSNR（ピーク信号対雑音比）を計算するための関数
from torchvision.utils import save_image  # 画像を保存するための関数
from yaml.constructor import ConstructorError  # YAMLのエラーを扱うためのクラス

class AttrDict(dict):
    """
    辞書のキーに属性のようにアクセスできる辞書クラスです。
    """
    __setattr__ = dict.__setitem__  # 属性の設定を辞書のアイテム設定に関連付け
    __getattr__ = dict.__getitem__  # 属性の取得を辞書のアイテム取得に関連付け

class Module(nn.Module):
    """
    レイヤーを遅延初期化し、再利用のためにキャッシュするカスタムPyTorchモジュールです。
    """
    def __init__(self):
        super().__init__()  # 親クラスの初期化
        self._modules = {}  # モジュールを保存するための辞書

    def get(self, name, PyTorch_layer, *args, **kwargs):
        """
        キャッシュされたレイヤーを取得し、存在しない場合は新しく作成します。
        """
        if name not in self._modules:  # 指定された名前が存在しない場合
            self._modules[name] = PyTorch_layer(*args, **kwargs)  # 新しくレイヤーを作成
        return self._modules[name]  # レイヤーを返す

class Step:
    """
    繰り返し回数を追跡するためのシンプルなカウンターです。
    """
    def __init__(self):
        self._step = nn.Parameter(torch.tensor(0), requires_grad=False)  # ステップカウンターを初期化

    def increment(self):
        self._step.data += 1  # ステップを1つ増やす

    def __call__(self):
        return self._step.item()  # 現在のステップを返す

def exp_name(cfg, model_dir_prefix=None):
    """
    設定に基づいて実験名を生成します。
    """
    exp_name = f"{cfg.dataset}_cwvae_{cfg.cell_type.lower()}"  # データセット名とモデルタイプ
    exp_name += f"_{cfg.levels}l_f{cfg.tmp_abs_factor}"  # 階層数と時間抽象化因子
    exp_name += f"_decsd{cfg.dec_stddev}"  # デコーダの標準偏差
    exp_name += f"_enchl{cfg.enc_dense_layers}_ences{cfg.enc_dense_embed_size}_edchnlmult{cfg.channels_mult}"  # エンコーダの設定
    exp_name += f"_ss{cfg.cell_stoch_size}_ds{cfg.cell_deter_size}_es{cfg.cell_embed_size}"  # セルのサイズ
    exp_name += f"_seq{cfg.seq_len}_lr{cfg.lr}_bs{cfg.batch_size}"  # シーケンスの長さ、学習率、バッチサイズ
    return exp_name  # 生成した実験名を返す

def validate_config(cfg):
    """
    設定を検証して互換性と整合性を確認します。
    """
    assert cfg.channels is not None and cfg.channels > 0, f"Incompatible channels = {cfg.channels} found in config."  # チャンネル数を確認
    assert cfg.open_loop_ctx % (cfg.tmp_abs_factor ** (cfg.levels - 1)) == 0, \
        f"Incompatible open-loop context length {cfg.open_loop_ctx} and temporal abstraction factor {cfg.tmp_abs_factor} for levels {cfg.levels}"  # コンテキストの整合性を確認
    assert cfg.datadir is not None, "data root directory cannot be None."  # データディレクトリがNoneでないことを確認
    assert cfg.logdir is not None, "log root directory cannot be None."  # ログディレクトリがNoneでないことを確認

def read_configs(config_path, base_config_path=None, **kwargs):
    """
    設定ファイル（YAML形式）を読み込んで解析し、設定辞書を返します。
    """
    class TorchDeviceLoader(yaml.SafeLoader):
        pass  # YAMLの安全なローダーを拡張

    def torch_device_constructor(loader, node):
        # 文字列形式を処理
        if isinstance(node, yaml.ScalarNode):
            device_str = loader.construct_scalar(node)  # スカラー値を取得
            return torch.device(device_str)  # torchデバイスオブジェクトを返す
        # リスト形式を処理
        elif isinstance(node, yaml.SequenceNode):
            sequence = loader.construct_sequence(node)  # シーケンスを取得
            if isinstance(sequence, list) and len(sequence) == 2 and sequence[0] == 'cuda':
                return torch.device(f'cuda:{sequence[1]}')  # CUDAデバイスを返す
        # どちらでもない場合はエラーを投げる
        raise ValueError(f"Invalid format for torch.device: {loader.construct_object(node)}")

    def attrdict_constructor(loader, node):
        return AttrDict(loader.construct_mapping(node, deep=True))  # AttrDictを生成

    # カスタムコンストラクターの登録
    TorchDeviceLoader.add_constructor('!torch.device', torch_device_constructor)
    TorchDeviceLoader.add_constructor('!attrdict', attrdict_constructor)

    if base_config_path is not None:  # 基本設定ファイルが指定されている場合
        base_config = yaml.safe_load(Path(base_config_path).read_text())  # 基本設定を読み込む
        config = base_config.copy()  # 基本設定をコピー
        config.update(yaml.load(Path(config_path).read_text(), Loader=TorchDeviceLoader))  # 新しい設定で更新
    else:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=TorchDeviceLoader)  # 設定ファイルを読み込む

    config = AttrDict(config)  # AttrDictに変換

    # 引数からデータディレクトリとログディレクトリを設定
    if kwargs.get("datadir", None) is not None:
        config.datadir = kwargs["datadir"]
    if kwargs.get("logdir", None) is not None:
        config.logdir = kwargs["logdir"]

    validate_config(config)  # 設定を検証
    return config  # 設定辞書を返す

def scan(cell, inputs, use_obs, initial):
    """
    入力シーケンスをスキャンし、各タイムステップでRNNセルを適用します。
    """
    assert initial is not None, "initial cannot be None. Pass zero_state instead."  # 初期状態がNoneでないことを確認
    
    outputs = []  # 出力を格納するリスト
    state = initial  # 初期状態を設定

    for t in range(inputs.size(1)):  # シーケンス長に従って反復
        inp_t = inputs[:, t]  # 時刻tの入力を取得
        out, state = cell(state, inp_t, use_obs=use_obs)  # セルを適用
        outputs.append(out)  # 出力をリストに追加

    outputs = torch.stack(outputs, dim=1)  # 出力を(B, T, ...)の形にスタック
    return outputs, state  # 出力と最終状態を返す

def _to_padded_strip(images):
    """
    画像のシーケンスを単一のパディングされたストリップに変換します。
    """
    if len(images.shape) <= 3:
        images = np.expand_dims(images, -1)  # チャンネル次元を追加
    c_dim = images.shape[-1]  # チャンネルの次元を取得
    x_dim = images.shape[-3]  # 高さの次元を取得
    y_dim = images.shape[-2]  # 幅の次元を取得
    padding = 1  # パディングのサイズ
    result = np.zeros(
        (x_dim, y_dim * images.shape[0] + padding * (images.shape[0] - 1), c_dim),  # パディングされた結果を初期化
        dtype=np.uint8,
    )
    for i in range(images.shape[0]):  # 各画像に対して
        result[:, i * y_dim + i * padding : (i + 1) * y_dim + i * padding, :] = images[i]  # 結果に画像を追加
    if result.shape[-1] == 1:  # グレースケールの場合
        result = np.reshape(result, result.shape[:2])  # チャンネル次元を削除
    return result  # 結果を返す

def compute_metrics(gt, pred):
    """
    グラウンドトゥルース (gt) と予測 (pred) のSSIMおよびPSNRを計算する関数
    """
    seq_len = gt.shape[1]  # シーケンスの長さを取得
    ssim = np.zeros((gt.shape[0], seq_len))  # SSIMを保存する配列
    psnr = np.zeros((gt.shape[0], seq_len))  # PSNRを保存する配列
    
    for i in range(gt.shape[0]):  # 各サンプルに対して
        for t in range(seq_len):  # 各タイムステップに対して
            # SSIMを計算
            ssim[i, t] = ssim_metric(
                gt[i, t], pred[i, t],
                multichannel=True,  # マルチチャンネル画像の場合
                data_range=gt[i, t].max() - gt[i, t].min(),  # データの範囲を指定
                win_size=5,  # ウィンドウサイズを指定
                channel_axis=-1  # チャンネルが最後の軸
            )
            # PSNRを計算
            psnr[i, t] = 10 * np.log10(1 / np.mean((gt[i, t] - pred[i, t]) ** 2))  # PSNRを計算
    
    return ssim, psnr  # SSIMとPSNRを返す

def plot_metrics(mean_metric, std_metric, logdir, metric_name):
    """
    指定されたメトリックの平均と標準偏差を時間にわたってプロットします。
    """
    x = np.arange(len(mean_metric))  # X軸の値を生成
    mean_metric = np.mean(mean_metric, axis=0)
    std_metric = np.std(mean_metric, axis=0)
    plt.figure()  # 新しい図を作成
    plt.plot(x, mean_metric, label=f"Mean {metric_name.upper()}")  # メトリックの平均をプロット
    plt.fill_between(x, mean_metric - std_metric, mean_metric + std_metric, alpha=0.2)  # 標準偏差の範囲を塗りつぶす
    plt.xlabel("Time Step")  # X軸のラベル
    plt.ylabel(metric_name.upper())  # Y軸のラベル
    plt.title(f"{metric_name.upper()} over Time")  # タイトルを設定
    plt.legend()  # 凡例を表示
    plt.grid(True)  # グリッドを表示
    plt.savefig(os.path.join(logdir, f"{metric_name}.png"))  # メトリックのプロットを保存
    plt.close()  # プロットを閉じる

def save_as_grid(images, full_save_path, nrow=8):
    """
    画像をグリッド形式で保存します。
    """
    try:
        save_image(images, full_save_path, nrow=nrow)  # 画像をグリッド形式で保存
    except Exception as e:
        print(f"[ERROR] Failed to save image grid to {full_save_path}: {str(e)}")  # エラーメッセージを表示
