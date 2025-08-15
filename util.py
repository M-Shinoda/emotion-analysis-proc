from pandas import DataFrame
from datetime import datetime, timedelta


def dataframe_to_jsonl(data: DataFrame):
    """
    DataFrameをJSONL形式に変換する関数

    Args:
        data (DataFrame): 入力データフレーム

    Returns:
        str: JSONL形式の文字列
    """
    return data.to_json(orient="records", lines=True)


def get_date_range(start_day, end_day):
    """
    開始日から終了日までの日付リストを取得する関数

    Args:
        start_day (str): 開始日 (例: "2025-08-01")
        end_day (str): 終了日 (例: "2025-08-14")

    Returns:
        list: 日付文字列のリスト
    """
    start_date = datetime.strptime(start_day, "%Y-%m-%d")
    end_date = datetime.strptime(end_day, "%Y-%m-%d")

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    return date_list
