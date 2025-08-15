import os
import json


def is_dev_environment():
    """
    環境変数 IS_DEV が True か False かを判定する関数

    Returns:
        bool: IS_DEV が "False" の場合は False、それ以外は True
    """
    return os.getenv("IS_DEV", "True").lower() == "true"


def get_service_account_key_path():
    """
    IS_DEV の値に応じて適切なサービスアカウントキーのパスを返す関数

    Returns:
        str: サービスアカウントキーのパス
    """
    if is_dev_environment():
        return "service_account_files/service-account-key-dev.json"
    else:
        return "service_account_files/service-account-key-prod.json"


def get_environment_type():
    """
    IS_DEV の値に応じて "prod" または "dev" を返す関数

    Returns:
        str: IS_DEV が True の場合は "dev"、それ以外は "prod"
    """
    return "dev" if is_dev_environment() else "prod"


def get_project_id_from_service_account():
    """
    IS_DEV の値に応じて適切なサービスアカウントキーの JSON ファイルから project_id を抽出して返す関数

    Returns:
        str: サービスアカウントキーの JSON ファイルに含まれる project_id
    """
    key_path = get_service_account_key_path()
    with open(key_path, "r") as f:
        service_account_data = json.load(f)
    return service_account_data.get("project_id")
