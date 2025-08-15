from pandas import DataFrame
from tqdm import tqdm
from bigquery import fetch_table_data, load_dataframe_to_bigquery
from emotionBert import calc_emotion_bert, convert_emotion_bert
from emotionLukeWrime import calc_emotion_luke_wrime
from query import (
    bert_emotion_data_query,
    luke_wrime_data_query,
    text_message_event_data_query,
    bert_emotion_table_id,
    luke_wrime_emotion_table_id,
)

from util import get_date_range
import warning  # ignore warning messages


def bert_analysis_by_day(day, live_data_day: DataFrame):
    """
    指定した日付のデータを取得し、感情分析を行い、BigQueryに保存する関数

    Args:
        day (str): データを取得する日付（YYYY-MM-DD形式）

    Returns:
        None
    """

    # ## BERTを用いた感情分析のためのデータ取得と処理
    # BigQueryからBERT Emotionデータを取得
    bert_emotion_data = fetch_table_data(bert_emotion_data_query(day))
    tqdm.write(f"▶ Number of BERT Emotion Data: {len(bert_emotion_data)}")

    # live_data_day と bert_emotion_data の id を比較して、
    # bert_emotion_data にない id の行を live_data_day から抽出
    bert_missing_ids_data = live_data_day[
        ~live_data_day["id"].isin(bert_emotion_data["id"])
    ]
    tqdm.write(f"▶ Number of missing IDs: {len(bert_missing_ids_data)}")

    # BERTを用いた感情分析の処理をし、JSONL形式に変換
    new_bert_data = convert_emotion_bert(bert_missing_ids_data, calc_emotion_bert)
    load_dataframe_to_bigquery(new_bert_data, bert_emotion_table_id)
    # ##


def luke_wrime_analysis_by_day(day, live_data_day: DataFrame):
    """
    指定した日付のデータを取得し、感情分析を行い、BigQueryに保存する関数

    Args:
        day (str): データを取得する日付（YYYY-MM-DD形式）

    Returns:
        None
    """

    # ## LUKE WRIMEを用いた感情分析のためのデータ取得と処理
    # BigQueryからLUKE WRIMEデータを取得
    luke_wrime_data = fetch_table_data(luke_wrime_data_query(day))
    tqdm.write(f"▶ Number of LUKE WRIME Data: {len(luke_wrime_data)}")

    # live_data_day と luke_wrime_data の id を比較して、
    # luke_wrime_data にない id の行を live_data_day から抽出
    luke_missing_ids_data = live_data_day[
        ~live_data_day["id"].isin(luke_wrime_data["id"])
    ]
    tqdm.write(f"▶ Number of missing IDs: {len(luke_missing_ids_data)}")

    # LUKE WRIMEを用いた感情分析の処理をし、JSONL形式に変換
    new_luke_data = convert_emotion_bert(luke_missing_ids_data, calc_emotion_luke_wrime)
    load_dataframe_to_bigquery(new_luke_data, luke_wrime_emotion_table_id)
    # ##


if __name__ == "__main__":
    days = get_date_range(
        # "2025-02-17", "2025-08-14"
        "2025-08-14",
        "2025-08-15",
    )  # 日付範囲を取得（単一日付の場合もリストで返す）

    for day in tqdm(days):
        tqdm.write(f"▶ 実行中: {day}")
        # BigQueryからLive Eventデータを取得
        live_data_day = fetch_table_data(text_message_event_data_query(day))
        tqdm.write(f"▶ Live Event Data Length: {len(live_data_day)}")

        bert_analysis_by_day(day, live_data_day)
        luke_wrime_analysis_by_day(day, live_data_day)
