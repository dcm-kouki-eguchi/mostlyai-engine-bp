# DataLoader最適化機能 仕様書

## 概要

mostlyai-engine-bpにおけるデータI/O速度改善のため、既存のPyTorch DataLoaderに最適化パラメータを追加する。CPUコア数に応じて自動的に最適なワーカー数を設定し、GPU転送やプリフェッチ機能を有効化してデータ読み込み性能を向上させる。

## 目的

- データ読み込み処理の並列化による高速化
- GPU転送の最適化
- プリフェッチによるレイテンシ削減
- 20-40%のデータI/O速度向上

## 機能仕様

### 1. 自動ワーカー数決定機能

#### 1.1 `get_optimal_num_workers()` 関数

**目的**: CPUコア数に応じて最適なワーカー数を自動決定

**入力パラメータ**: なし

**出力**: `int` - 最適なワーカー数

**決定ロジック**:
```python
cpu_cores = os.cpu_count()  # 論理コア数
physical_cores = psutil.cpu_count(logical=False)  # 物理コア数

# 物理コア数の50-75%程度が最適、最大4に制限
optimal_workers = min(4, max(1, physical_cores // 2))
```

**エラーハンドリング**:
- psutil利用不可時: `physical_cores = cpu_cores // 2` でフォールバック

### 2. DataLoaderパラメータ最適化

#### 2.1 追加パラメータ

既存のDataLoader作成部分に以下のパラメータを追加:

```python
trn_dataloader = DataLoader(
    dataset=trn_dataset,
    shuffle=True,
    batch_size=trn_batch_size if with_dp else batch_size,
    collate_fn=batch_collator,
    # 追加パラメータ
    num_workers=get_optimal_num_workers(),  # CPUコア数に応じて自動調整
    pin_memory=torch.cuda.is_available(),   # GPU転送の高速化
    prefetch_factor=2,                      # バッファリング
    persistent_workers=True,                # ワーカープロセス再利用
)
```

#### 2.2 各パラメータの説明

##### `num_workers`
- **役割**: データ読み込みを並列化するサブプロセス数
- **効果**: データ読み込みとモデル訓練を並列実行、I/Oボトルネックを解消
- **設定**: `get_optimal_num_workers()`で自動決定

##### `pin_memory`
- **役割**: CPUメモリをGPUが直接アクセス可能な領域に固定
- **効果**: CPU→GPU転送が20-30%高速化
- **設定**: `torch.cuda.is_available()`の結果に基づく

##### `prefetch_factor`
- **役割**: 各ワーカーが事前に準備するバッチ数
- **効果**: データ準備の待ち時間を削減、GPU処理中に次のバッチを準備
- **設定**: 固定値 `2`

##### `persistent_workers`
- **役割**: エポック間でワーカープロセスを再利用
- **効果**: プロセス起動コストを削減、初期化処理の重複を回避
- **設定**: 固定値 `True`

### 3. 既存コードとの統合

#### 3.1 training.py への統合

**変更箇所**: DataLoader作成部分

**変更前**:
```python
trn_dataloader = DataLoader(
    dataset=trn_dataset,
    shuffle=True,
    batch_size=trn_batch_size if with_dp else batch_size,
    collate_fn=batch_collator,
)
```

**変更後**:
```python
trn_dataloader = DataLoader(
    dataset=trn_dataset,
    shuffle=True,
    batch_size=trn_batch_size if with_dp else batch_size,
    collate_fn=batch_collator,
    # 追加パラメータ
    num_workers=get_optimal_num_workers(),
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=2,
    persistent_workers=True,
)
```

#### 3.2 検証用DataLoaderの最適化

**対象**: `val_dataloader` も同様に最適化
```python
val_dataloader = DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=val_batch_size,
    collate_fn=batch_collator,
    # 追加パラメータ
    num_workers=get_optimal_num_workers(),
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=2,
    persistent_workers=True,
)
```

### 4. パフォーマンス期待値

#### 4.1 データサイズ別の改善効果

- **小規模データ** (< 1GB): 10-20%の高速化
- **中規模データ** (1-10GB): 20-40%の高速化
- **大規模データ** (> 10GB): 30-50%の高速化

#### 4.2 データタイプ別の効果

- **フラットデータ**: 標準的な改善効果
- **シーケンシャルデータ**: より顕著な改善効果（パディング処理の並列化）

### 5. ログ出力仕様

#### 5.1 最適化パラメータのログ

```
INFO: CPU cores: 8 logical, 4 physical
INFO: Optimal num_workers: 2
```

#### 5.2 システム情報のログ

- 論理CPUコア数
- 物理CPUコア数
- 最適ワーカー数

### 6. エラーハンドリング

#### 6.1 psutil利用不可時のフォールバック

```python
try:
    physical_cores = psutil.cpu_count(logical=False)
except Exception:
    physical_cores = cpu_cores // 2  # フォールバック
```

### 7. 互換性

#### 7.1 既存機能との互換性

- 既存のバッチサイズヒューリスティック機能は維持
- 差分プライバシー機能との完全互換性
- 既存のBatchCollator機能は変更なし

#### 7.2 プラットフォーム互換性

- Linux/Windows/macOS対応
- GPU有無に関わらず動作
- Python multiprocessing対応

### 8. 実装上の注意点

#### 8.1 Windowsでの制限事項

- `num_workers > 0` の場合、`if __name__ == '__main__':` が必要
- 既存コードでは問題なし（ライブラリ内での使用のため）

#### 8.2 メモリ使用量の考慮

- ワーカープロセス数に比例してメモリ使用量が増加
- シーケンシャルデータでは特に注意が必要

### 9. 実装手順

#### 9.1 `get_optimal_num_workers()` 関数の追加

training.pyに関数を追加

#### 9.2 DataLoader作成部分の修正

既存の2箇所のDataLoader作成部分にパラメータを追加
- `trn_dataloader`
- `val_dataloader`

## まとめ

この仕様に基づいてDataLoader最適化機能を実装することで、mostlyai-engine-bpのデータI/O性能を大幅に改善し、ユーザーの学習時間短縮に貢献する。自動最適化により、ユーザーは手動調整なしに最適な性能を得ることができる。
