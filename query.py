from envManager import (
    get_environment_type,
    get_project_id_from_service_account,
)


dataset_id_no_suffix = (
    "youtube_c7_kqMFDE8c_"  # データセットIDのデプロイ環境に依存しない部分
)

project_id = get_project_id_from_service_account()
dataset_id = f"{dataset_id_no_suffix}{get_environment_type()}"

live_event_table_id = "live_event"
bert_emotion_table_id = "bert_emotion"
luke_wrime_emotion_table_id = "luke_wrime_emotion"


def text_message_event_data_query(day):
    get_text_message_event_data_query = f"""
    SELECT id, snippet_publishedAt, snippet_displayMessage
    FROM `{project_id}.{dataset_id}.{live_event_table_id}`
    WHERE 
        TIMESTAMP_TRUNC(snippet_publishedAt, DAY) = TIMESTAMP("{day}") AND
        snippet_type = "textMessageEvent"
    """
    return get_text_message_event_data_query


def bert_emotion_data_query(day):
    get_bert_emotion_data_query = f"""
    SELECT id
    FROM `{project_id}.{dataset_id}.{bert_emotion_table_id}`
    WHERE TIMESTAMP_TRUNC(publishedAt, DAY) = TIMESTAMP("{day}")
    """
    return get_bert_emotion_data_query


def luke_wrime_data_query(day):
    get_luke_wrime_data_query = f"""
    SELECT id
    FROM `{project_id}.{dataset_id}.{luke_wrime_emotion_table_id}`
    WHERE TIMESTAMP_TRUNC(publishedAt, DAY) = TIMESTAMP("{day}")
    """
    return get_luke_wrime_data_query
