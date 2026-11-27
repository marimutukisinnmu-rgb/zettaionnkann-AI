# 絶対音感AI

音声ファイルから単音の音名とMIDIノート番号を推定するPythonプロジェクトです。

## v0.4.0 の方針

v0.4はディープラーニングを使用しません。FFT・倍音解析・YINなどのDSPに、scikit-learnのExtraTrees候補再ランキングを組み合わせます。

優先順位：

1. **実用音域 MIDI 24〜95**：正解率と処理速度を最優先
2. **低音域 MIDI 1〜23**：基音推定とオクターブ誤判定を重点改善
3. **高音域 MIDI 96〜127**：対応を維持し、他の音域への悪影響を避ける

MIDI 0〜127への対応は維持します。現在のベンチマークはMIDI音源でレンダリングしたWAVを使用しており、実際の生楽器録音への精度を保証するものではありません。

## ファイル構成

- main.py：CLI入口。従来の python main.py / python main.py "test.wav" を維持
- pitch_dsp_v040.py：FFT、倍音、YIN、CQT、候補生成などのDSP
- pitch_model_v040.py：合成データ生成とExtraTrees候補再ランキングモデルの学習
- pitch_runtime_v040.py：Adaptive解析、LOW_CONF、ファイル解析結果
- benchmark.py：SKIP対応・日時別保存のベンチマーク
- models/pitch_reranker_v040.pkl：v0.4専用の学習済みモデル（初回学習後に作成）

## v0.4 の解析方式

- **FFT**：共通スペクトルを作る中核処理。倍音ピークをまとめて評価します。
- **倍音テンプレート**：基音候補と倍音列の整合性を評価し、倍音関連の候補も生成します。
- **YIN**：差分関数とCMNDに基づいて周期・基音候補を推定します。CMNDを信頼度の目安として保持します。
- **自己相関**：周期情報を補助的に利用します。
- **CQT**：Tier 0で判断が確定しない場合に計算し、候補生成と候補スコアの補助に使います。
- **Attack / Sustain**：さらに判断が不確実な場合だけ別々に解析します。
- **ExtraTrees候補再ランキング**：各解析器が作った候補を評価して並べ替えます。v0.3の全128音直接分類とは異なります。
- **LOW_CONF**：候補の差が十分でない場合、判定を保留して上位候補を表示します。
- **Adaptive解析**：最初から全解析器を動かさず、信頼できる一致があれば早期終了します。

### Adaptive の段階

| Tier | 実行内容 | 用途 |
|---|---|---|
| 0 | 入力チェック、FFT、倍音、YIN・自己相関 | 複数の証拠が一致した場合は早期終了 |
| 1 | CQT、候補生成、ExtraTrees再ランキング | Tier 0で決まらない音を検証 |
| 2 | 複数区間のYIN、Attack、Sustain | さらに曖昧な場合の追加検証 |

Tier 0の早期終了は、YIN・FFT・倍音候補が厳しい条件で一致した場合に限ります。Tier 1ではCQTを補助スコアとして使用し、Tier 2では時間変化と複数フレームの周期推定を追加します。

## 判定ステータス

- OK：通常の確信度で音高を推定
- LOW_CONF：候補はあるが十分な確信がない。最有力候補と上位候補を表示
- SKIP：無音・空データ・有効区間が短すぎるなど、音声として解析できない
- ERROR：ファイル読み込みや解析処理そのものの異常

SKIPの場合は次のメッセージを表示します。

    有効な音声ではありません

無音判定は、ほぼ無音の信号と50 ms未満の有効区間などを検出する保守的なゲートです。小さな音量で録音された音声を誤ってSKIPしないよう、固定の大きなdBFS閾値は採用していません。

LOW_CONFのConfidence scoreは候補間の相対スコア差から計算するヒューリスティック値です。較正済みの正解確率ではありません。

## 動作環境

Python 3.10以降を推奨します。GPUやCUDAは不要です。v0.4はCPUで動くExtraTreesを使用し、PyTorchなどのディープラーニング用ライブラリは使用しません。

## インストール

    python -m pip install numpy librosa scikit-learn joblib soundfile

## 初回起動・学習

    python main.py

v0.4の候補再ランキングモデルが存在しない場合、128音それぞれの合成音を使って学習し、models/pitch_reranker_v040.pkl に保存します。

モデルを作り直す場合：

    python main.py --retrain

v0.3のモデル models/pitch_ai.pkl は削除・上書きしません。v0.4は別のモデルファイルを使用するため、v0.4用の初回学習が必要です。

## 音声ファイルを判定

    python main.py "test.wav"

出力には、最終候補、MIDI番号、周波数、ステータス、相対Confidence score、解析Tier、各解析器の判定、上位候補、処理時間が含まれます。

## ベンチマーク

既定のWAVフォルダ：

    E:\rise-\temp\midi音\wav

ファイル名形式：

    (楽器番号) - (Notes番号) - (音名).wav

例：

    000 - 001 - C#-1.wav
    000 - 002 - D-1.wav

実行：

    python benchmark.py

別のWAVフォルダを指定する場合：

    python benchmark.py --wav-dir "E:\path\to\wav"

各実行は bench_runs\v0.4.0_日時\ に保存されます。過去の実行結果やv0.3のログを上書きしません。

出力ファイル：

- benchmark_results.txt：コンソールと同じ詳細ログ
- results.jsonl：1ファイル1行の構造化結果
- summary.json：全体・音域別・楽器別の集計と処理時間
- input_manifest.csv：対象ファイル名・期待音名・サイズ・更新日時

ベンチマークでは次を別々に集計します。

- OK / NG / LOW_CONF / SKIP / ERROR
- 全ファイル分母の正解率
- SKIPを除く正解率（ERRORは失敗として分母に含む）
- 音域別・楽器別の正解率
- オクターブ違い・半音違い・倍音ロック・その他の誤り
- 平均・中央値・p95の処理時間、全体の壁時間とCPU時間

LOW_CONFは最有力候補で正誤を採点しつつ、LOW_CONF件数として別に表示します。SKIPは誤判定数に含めず、入力カバレッジと全ファイル分母の正解率に反映します。

## 注意・既知の制約

- 主な対象は単音の推定です。和音・複数音を完全に分離して採譜する機能ではありません。
- 学習データは音色・倍音構成を変化させた合成音です。実楽器や未知の音色に対する性能は別途検証が必要です。
- 候補再ランキングモデルは合成データで学習します。候補生成の網羅率と、実WAVベンチマークでの正解率は別の指標です。
- LOW_CONFの相対Confidence scoreは較正済み確率ではありません。
- CQT・Attack・Sustainの追加解析は入力によっては処理時間を増やします。Tier 0で高信頼の一致がある場合は追加処理を省略します。
- 014・016など一部の楽器では、ラベルやレンダリングされた音高の確認が必要な可能性があります。疑わしいデータを自動的に除外せず、元WAVを別途検証してください。


## v0.4.1 candidate-generation experiment

v0.4.1 is an experimental DSP update that preserves the v0.4.0 ExtraTrees
model and its 23-feature input schema.

Changes:
- Adds up to 8 prominent local FFT peaks as supplemental MIDI candidates.
- Does not force MIDI 127 into the candidate set; G9 is added only when a
  qualifying spectral peak is actually present.
- YIN abstains when no local CMND minimum passes the configured threshold.
- Reuses `models/pitch_reranker_v040.pkl`; no model retraining or overwrite.

Run:
```powershell
python main_v041.py "test.wav"
python benchmark_v041.py
```

This is an experimental branch of the pipeline. Compare its coverage and
accuracy against the preserved v0.4.0 benchmark before treating it as an
improvement. The spectral peak supplement can include harmonics, so the
reranker still has to distinguish fundamentals from partials.


## v0.4.2 high-register DSP update (experimental)

v0.4.2 focuses on ordinary pitched GM Bank 0 programs 0-119. GM Bank 0
programs 120-127 are sound effects and are excluded from this version's
benchmark. Percussion Bank 128 is deferred.

Changes:
- Adds a separate high-register YIN pass for MIDI 96-127, triggered only
  when the corresponding high-frequency spectrum contains sufficient energy.
- Reserves part of the supplemental spectral-peak candidate budget for
  MIDI 96-127 so low-frequency peaks cannot consume every slot.
- Uses log-parabolic sub-bin peak interpolation.
- Adds cents-tolerant fundamental, upper-harmonic, subharmonic, and
  high-band-YIN evidence to the candidate feature vector (23 -> 28 features).
- Trains an independent ExtraTrees model at
  models/pitch_reranker_v042.pkl. Existing v0.4.0/v0.4.1 model files are not
  overwritten.
- Fixes the benchmark scope to GM Bank 0 programs 0-119 and records excluded
  program counts in summary.json. Note labels still cover MIDI 0-127.

Run:
```powershell
python main_v042.py "test.wav"
python benchmark_v042.py
```

The benchmark uses the same WAV naming convention but excludes program
numbers 120-127 (sound effects). Run the 0-119 subset on the same files and
compare it with the corresponding subset of prior versions; do not compare
the filtered total directly with the full 128-program v0.4.1 total.

This implementation is experimental until local compilation, a small
smoke test, model training, and the full benchmark complete successfully.
