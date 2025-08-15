import pandas as pd
import numpy as np


def diff_report(
    df1: pd.DataFrame, df2: pd.DataFrame, key=None, float_rtol=1e-5, float_atol=1e-8
):
    """
    2つのDataFrameの違いをまとめて返す。
    - key を指定すると、そのキーで行を突き合わせて「値の変更」をセル単位で出す
    - key が無い場合は、行の追加/削除のみを検出（値変更は「削除+追加」とみなされる）
    戻り値: dict of DataFrame
      - cols_only_in_df1 / cols_only_in_df2 / dtype_diff
      - rows_only_in_df1 / rows_only_in_df2
      - value_diff（key を指定したときのみ）
    """
    out = {}

    # 1) スキーマ差分
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    out["cols_only_in_df1"] = pd.Index(sorted(cols1 - cols2))
    out["cols_only_in_df2"] = pd.Index(sorted(cols2 - cols1))
    common_cols = sorted(cols1 & cols2)

    if common_cols:
        d1 = df1[common_cols].dtypes.astype(str)
        d2 = df2[common_cols].dtypes.astype(str)
        dtype_diff_mask = d1 != d2
        out["dtype_diff"] = pd.DataFrame(
            {"df1": d1[dtype_diff_mask], "df2": d2[dtype_diff_mask]}
        )
    else:
        out["dtype_diff"] = pd.DataFrame(columns=["df1", "df2"])

    # 2) 行の存在差分
    if key is None:
        # 全共通列で完全一致する行を基準に「片方だけ」を検出
        common = common_cols
        if not common:
            # 完全に別物
            out["rows_only_in_df1"] = df1.copy()
            out["rows_only_in_df2"] = df2.copy()
            return out
        # 重複を考慮して一意キーを付与
        df1_ = df1[common].copy()
        df2_ = df2[common].copy()
        df1_["_count"] = 1
        df2_["_count"] = 1
        g1 = df1_.groupby(common)["_count"].sum().rename("n1")
        g2 = df2_.groupby(common)["_count"].sum().rename("n2")
        cmp = pd.concat([g1, g2], axis=1).fillna(0).astype(int)
        only1_idx = cmp.index.repeat((cmp["n1"] - cmp["n2"]).clip(lower=0))
        only2_idx = cmp.index.repeat((cmp["n2"] - cmp["n1"]).clip(lower=0))
        out["rows_only_in_df1"] = pd.DataFrame(list(only1_idx), columns=common)
        out["rows_only_in_df2"] = pd.DataFrame(list(only2_idx), columns=common)
        return out

    # key あり：追加・削除・値変更を出す
    key = [key] if isinstance(key, str) else list(key)
    # 存在しないキーがあれば落とす
    missing = [k for k in key if k not in df1.columns or k not in df2.columns]
    if missing:
        raise KeyError(f"Key not found in both: {missing}")

    df1k = df1.set_index(key)
    df2k = df2.set_index(key)

    added_keys = df2k.index.difference(df1k.index)
    removed_keys = df1k.index.difference(df2k.index)
    out["rows_only_in_df2"] = df2k.loc[added_keys].reset_index()
    out["rows_only_in_df1"] = df1k.loc[removed_keys].reset_index()

    # 値の差分（共通キーのみ・共通列のみ）
    common_idx = df1k.index.intersection(df2k.index)
    common_cols_for_values = [c for c in common_cols if c not in key]
    a = df1k.loc[common_idx, common_cols_for_values].copy()
    b = df2k.loc[common_idx, common_cols_for_values].copy()

    # 浮動小数点は許容誤差で等価とみなす
    diff_mask = pd.DataFrame(False, index=common_idx, columns=common_cols_for_values)
    for c in common_cols_for_values:
        s1, s2 = a[c], b[c]
        if np.issubdtype(s1.dtype, np.floating) and np.issubdtype(
            s2.dtype, np.floating
        ):
            eq = np.isclose(
                s1.astype(float),
                s2.astype(float),
                rtol=float_rtol,
                atol=float_atol,
                equal_nan=True,
            )
        else:
            eq = (s1 == s2) | (s1.isna() & s2.isna())
        diff_mask[c] = ~eq

    if diff_mask.any().any():
        # “縦長”に展開：key + col + df1 + df2
        where = diff_mask.stack()
        idx_cols = list(a.index.names)
        diffs = (
            pd.DataFrame({"_diff": where})
            .query("_diff")
            .drop(columns="_diff")
            .reset_index()
            .rename(
                columns={
                    "level_0": idx_cols[0] if idx_cols else "index",
                    "level_1": "column",
                }
            )
        )
        # 値を付与
        diffs["df1"] = [
            a.loc[tuple(r[idx_cols]) if idx_cols else r[idx_cols[0]], r["column"]]
            for _, r in diffs.iterrows()
        ]
        diffs["df2"] = [
            b.loc[tuple(r[idx_cols]) if idx_cols else r[idx_cols[0]], r["column"]]
            for _, r in diffs.iterrows()
        ]
    else:
        diffs = pd.DataFrame(columns=key + ["column", "df1", "df2"])

    out["value_diff"] = diffs
    return out
