import torch
import pandas as pd
from pandas import DataFrame
from collections.abc import Callable
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig

tokenizer = AutoTokenizer.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime"
)
config = LukeConfig.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
    output_hidden_states=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
    config=config,
)
max_seq_length = 512

tqdm.pandas()  # tqdmをpandasに統合


# def calc_emotion_luke_wrime_demo(text: str) -> tuple:  # (label, score)
#     # ここでは、テキストの最初の4文字をラベルとし、スコアはテキストの長さの10%を四捨五入して返す
#     label = text[:4]
#     score = round(len(text) * 0.1, 2)
#     return label, score


def calc_emotion_luke_wrime(text: str) -> tuple:  # (label, score)
    global model

    token = tokenizer(
        text, truncation=True, max_length=max_seq_length, padding="max_length"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # モデルを適切なデバイスに移動

    input_ids = torch.tensor(token["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(token["attention_mask"]).unsqueeze(0).to(device)

    output = model(input_ids, attention_mask)
    max_index = torch.argmax(output.logits, dim=1).item()
    index = output.logits.cpu().detach().numpy()  # 必要に応じてCPUに戻す
    return max_index, index


def convert_emotion_luke_wrime(
    data: DataFrame, function: Callable[[str], tuple]
) -> DataFrame:
    """
    LUKEを用いた感情分析の処理を模擬し、適当なDataFrameを返す関数

    Args:
        data (DataFrame): 入力データフレーム

    Returns:
        DataFrame: 感情分析結果を含むデータフレーム
    """

    if data.empty:
        return pd.DataFrame()

    # データフレームをコピー
    result_data = data.copy()

    def process_row(text):
        max_sentiment_index, sentiment_index = function(text)
        torch.set_printoptions(
            precision=17,  # 有効桁数（float64なら最大17桁程度）
            threshold=float("inf"),  # 要素数が多くても省略しない
            linewidth=200,  # 1行に表示する最大幅（改行を防ぐ）
            sci_mode=False,  # 科学的記法（指数表記）を使わない
        )

        return pd.Series(
            {
                "luke_wrime_score_joy": float(sentiment_index[0][0]),
                "luke_wrime_score_sadness": float(sentiment_index[0][1]),
                "luke_wrime_score_anticipation": float(sentiment_index[0][2]),
                "luke_wrime_score_surprise": float(sentiment_index[0][3]),
                "luke_wrime_score_anger": float(sentiment_index[0][4]),
                "luke_wrime_score_fear": float(sentiment_index[0][5]),
                "luke_wrime_score_disgust": float(sentiment_index[0][6]),
                "luke_wrime_score_trust": float(sentiment_index[0][7]),
                "luke_wrime_index": int(max_sentiment_index),
            }
        )

    # 各行のsnippet_displayMessageを処理し、結果を新しいデータフレームに格納
    processed_data = result_data["snippet_displayMessage"].progress_apply(process_row)

    # 元のデータフレームに計算結果を結合
    result_data = pd.concat([result_data, processed_data], axis=1)

    result_data = result_data.drop(columns=["snippet_displayMessage"])  # 不要な列を削除
    result_data = result_data.rename(
        columns={"snippet_publishedAt": "publishedAt"}
    )  # 列名を変更
    result_data["luke_wrime_index"] = result_data["luke_wrime_index"].astype(
        int
    )  # 型変換

    return result_data
