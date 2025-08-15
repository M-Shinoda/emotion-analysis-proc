import os
import unittest
from emotionBert import convert_emotion_bert, calc_emotion_bert_demo, calc_emotion_bert
from emotionLukeWrime import calc_emotion_luke_wrime, convert_emotion_luke_wrime
from envManager import is_dev_environment
import pandas as pd


class TestIsLocalEnvironment(unittest.TestCase):
    def test_is_dev_true(self):
        """IS_DEV が 'True' の場合のテスト"""
        os.environ["IS_DEV"] = "True"
        self.assertTrue(is_dev_environment())

    def test_is_dev_false(self):
        """IS_DEV が 'False' の場合のテスト"""
        os.environ["IS_DEV"] = "False"
        self.assertFalse(is_dev_environment())

    def test_is_dev_not_set(self):
        """IS_DEV が設定されていない場合のテスト"""
        if "IS_DEV" in os.environ:
            del os.environ["IS_DEV"]
        self.assertTrue(is_dev_environment())


class TestDataframeToEmotionBert(unittest.TestCase):
    def test_dataframe_to_emotion_bert(self):
        """DataFrameをEmotionBert形式のDataFrameに変換するテスト"""

        # サンプルデータフレームを作成
        data = {
            "id": [1, 2, 3],
            "snippet_publishedAt": [
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
            ],
            "snippet_displayMessage": ["HOGE", "HUGE", "HOGEHUGE"],
        }
        destination_data = {
            "id": [1, 2, 3],
            "publishedAt": [
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
            ],
            "label": ["HOGE", "HUGE", "HOGE"],
            "score": [0.4, 0.4, 0.8],  # 各文字列の長さをスコアとする
        }

        df = pd.DataFrame(data)
        destination_df = pd.DataFrame(destination_data)

        result_df = convert_emotion_bert(df, calc_emotion_bert_demo)

        print()
        print("DataFrameをEmotionBert形式のDataFrameに変換するテスト")
        print("結果データフレーム:")
        print(result_df)
        print("期待されるデータフレーム:")
        print(destination_df)

        # result_df と destination_df が一致するかを確認
        self.assertTrue(result_df.equals(destination_df))


class TestDataframeToJsonl(unittest.TestCase):
    def test_dataframe_to_jsonl(self):
        """DataFrameをJSONL形式に変換するテスト"""
        import pandas as pd
        from util import dataframe_to_jsonl

        # サンプルデータフレームを作成
        data = {
            "id": [1, 2, 3],
            "publishedAt": [
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
            ],
            "label": ["HOGE", "HUGE", "HOGE"],
            "score": [0.4, 0.4, 0.8],
        }

        df = pd.DataFrame(data)

        # 期待されるJSONL文字列
        expected_jsonl = """\
{"id":1,"publishedAt":"2025-08-14T05:54:34.042904+00:00","label":"HOGE","score":0.4}\n\
{"id":2,"publishedAt":"2025-08-14T05:54:34.042904+00:00","label":"HUGE","score":0.4}\n\
{"id":3,"publishedAt":"2025-08-14T05:54:34.042904+00:00","label":"HOGE","score":0.8}
"""

        # 関数を使用してJSONL形式に変換
        result_jsonl = dataframe_to_jsonl(df)
        print()
        print("DataFrameをJSONL形式に変換するテスト")
        print("結果JSONL:")
        print(result_jsonl)
        print("期待されるJSONL:")
        print(expected_jsonl)

        # 結果が期待されるJSONLと一致するかを確認
        self.assertEqual(result_jsonl, expected_jsonl)


class TestBertEmotionAnalysis(unittest.TestCase):
    """BERTを用いた感情分析のテスト"""

    def test_convert_emotion_bert(self):

        import pandas as pd
        from util import dataframe_to_jsonl

        # サンプルデータフレームを作成
        data = {
            "id": [1, 2, 3],
            "snippet_publishedAt": [
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
            ],
            "snippet_displayMessage": ["こんにちは", "怖いです", "希望が持てます"],
        }

        df = pd.DataFrame(data)

        destination_data = {
            "id": [1, 2, 3],
            "publishedAt": [
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
            ],
            "label": ["NEUTRAL", "NEGATIVE", "POSITIVE"],  # ラベルは固定値
            "score": [
                0.9328477382659912,
                0.9943292737007141,
                0.8055524826049805,
            ],  # スコアは固定値
        }

        destination_df = pd.DataFrame(destination_data)
        result_df = convert_emotion_bert(df, calc_emotion_bert)

        self.assertTrue(result_df.equals(destination_df))


class TestLukeWrimeEmotionAnalysis(unittest.TestCase):
    """LUKEを用いた感情分析のテスト"""

    def test_convert_emotion_luke_wrime(self):

        import pandas as pd

        # サンプルデータフレームを作成
        data = {
            "id": [1, 2, 3],
            "snippet_publishedAt": [
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
            ],
            "snippet_displayMessage": ["こんにちは", "怖いです", "希望が持てます"],
        }

        df = pd.DataFrame(data)

        destination_data = {
            "id": [1, 2, 3],
            "publishedAt": [
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
                "2025-08-14T05:54:34.042904+00:00",
            ],
            "luke_wrime_score_joy": [
                3.97590494155883789,
                -4.48982763290405273,
                -3.84145450592041016,
            ],
            "luke_wrime_score_sadness": [
                -5.58582353591918945,
                -4.04886293411254883,
                -4.56968736648559570,
            ],
            "luke_wrime_score_anticipation": [
                -4.59427213668823242,
                -4.25647258758544922,
                3.16512489318847656,
            ],
            "luke_wrime_score_surprise": [
                -5.92830705642700195,
                -3.52139067649841309,
                -6.16881132125854492,
            ],
            "luke_wrime_score_anger": [
                -5.75389575958251953,
                -4.64302444458007812,
                -6.47149181365966797,
            ],
            "luke_wrime_score_fear": [
                -5.66050100326538086,
                2.75487995147705078,
                -5.35040473937988281,
            ],
            "luke_wrime_score_disgust": [
                -4.99225521087646484,
                -4.72713661193847656,
                -6.40688037872314453,
            ],
            "luke_wrime_score_trust": [
                -5.22053527832031250,
                -5.63708782196044922,
                -6.22049236297607422,
            ],
            "luke_wrime_index": [0, 5, 2],
        }

        destination_df = pd.DataFrame(destination_data)
        result_df = convert_emotion_luke_wrime(df, calc_emotion_luke_wrime)
        # pd.set_option("display.precision", 17)  # 小数点以下の表示桁数
        # pd.set_option("display.float_format", "{:.17f}".format)  # フォーマット固定
        # pd.set_option("display.max_rows", None)  # 行の省略を防ぐ
        # pd.set_option("display.max_columns", None)  # 列の省略を防ぐ
        # pd.set_option("display.width", None)  # 自動改行を防ぐ

        print("結果データフレーム:")
        print(result_df)
        print("期待されるデータフレーム:")
        print(destination_df)

        self.assertTrue(result_df.equals(destination_df))


if __name__ == "__main__":
    unittest.main()
