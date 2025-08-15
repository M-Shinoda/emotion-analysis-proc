from google.cloud import bigquery
from pandas import DataFrame

from util import dataframe_to_jsonl
from envManager import get_service_account_key_path
from query import project_id, dataset_id

# クライアントの初期化
service_account_key_path = get_service_account_key_path()
client = bigquery.Client.from_service_account_json(service_account_key_path)


def fetch_table_data(query=None) -> DataFrame:
    """
    サービスアカウントを使用してBigQueryからデータを取得する関数

    Args:
        query (str): 実行するクエリ。Noneの場合は全データを取得

    Returns:
        pandas.DataFrame: テーブルデータ
    """

    # クエリの実行
    query_job = client.query(query)

    # 結果をデータフレームとして取得
    result = query_job.result()
    result_dataframe = result.to_dataframe()
    result_dataframe = format_datetime_column(result_dataframe, "snippet_publishedAt")

    return result_dataframe


def load_dataframe_to_bigquery(dataframe: DataFrame, table_id: str) -> None:
    """
    DataFrameをBigQueryに書き込む関数

    Args:
        dataframe (pandas.DataFrame): 書き込むデータフレーム
        table_id (str): 書き込み先のBigQueryテーブルID
            (例: "project_id.dataset_id.table_id")

    Returns:
        None
    """
    from io import StringIO

    if dataframe.empty:
        print("DataFrame is empty. No data to load.")
        return

    jsonl_data = dataframe_to_jsonl(dataframe)

    # JSONLデータをStringIOに変換
    jsonl_stream = StringIO(jsonl_data)

    # BigQueryにロード
    job = client.load_table_from_file(
        jsonl_stream,
        f"{project_id}.{dataset_id}.{table_id}",
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            autodetect=True,  # 自動でスキーマ推定（指定も可能）
        ),
    )

    # ジョブの完了を待機
    job.result()


def format_datetime_column(dataframe: DataFrame, column_name: str) -> DataFrame:
    """
    指定された列の datetime64[us, UTC] データを ISO 8601 フォーマットに変換する関数

    Args:
        dataframe (pandas.DataFrame): 対象のデータフレーム
        column_name (str): 対象の列名
    Returns:
        pandas.DataFrame: フォーマットが適用されたデータフレーム
    """
    if column_name in dataframe.columns:
        dataframe[column_name] = dataframe[column_name].dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f%z"
        )
    return dataframe
