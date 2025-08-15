import torch
import pandas as pd
from pandas import DataFrame
from collections.abc import Callable
from tqdm import tqdm

from transformers import pipeline
from transformers import (
    AutoModelForSequenceClassification,
    BertJapaneseTokenizer,
)

model = AutoModelForSequenceClassification.from_pretrained(
    "koheiduck/bert-japanese-finetuned-sentiment"
)
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def calc_emotion_bert_demo(text: str) -> tuple:  # (label, score)
    # ここでは、テキストの最初の4文字をラベルとし、スコアはテキストの長さの10%を四捨五入して返す
    label = text[:4]
    score = round(len(text) * 0.1, 2)
    return label, score


def calc_emotion_bert(text: str) -> tuple:  # (label, score)
    global model
    global classifier

    # GPUを指定した場合は、処理が高速になる
    if torch.cuda.is_available():
        model = model.to("cuda")

    result = classifier(text)[0]
    return result["label"], result["score"]


def convert_emotion_bert(
    data: DataFrame, function: Callable[[str], tuple]
) -> DataFrame:
    """
    BERTを用いた感情分析の処理を模擬し、適当なDataFrameを返す関数

    Args:
        data (DataFrame): 入力データフレーム

    Returns:
        DataFrame: 感情分析結果を含むデータフレーム
    """

    if data.empty:
        return pd.DataFrame()

    # データフレームをコピー
    result_data = data.copy()

    # 各行に対して処理を加える
    result_data["label"], result_data["score"] = zip(
        *[
            function(x)
            for x in tqdm(result_data["snippet_displayMessage"], desc="Processing rows")
        ]
    )

    result_data = result_data.drop(columns=["snippet_displayMessage"])  # 不要な列を削除
    result_data = result_data.rename(
        columns={"snippet_publishedAt": "publishedAt"}
    )  # 列名を変更

    return result_data
