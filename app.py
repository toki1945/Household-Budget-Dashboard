from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REQUIRED_COLUMNS = {
    "計算対象",
    "日付",
    "金額（円）",
    "大項目",
    "中項目",
    "振替",
}
DETAIL_COLUMNS = [
    "日付",
    "内容",
    "金額（円）",
    "大項目",
    "中項目",
    "保有金融機関",
    "計算対象",
    "振替",
    "元ファイル",
]
# 予測に必要な最小月数（学習用の履歴月数。これに当月を加えた月数のデータが必要）
MINIMUM_HISTORY_MONTHS = 3
JST = "Asia/Tokyo"


@st.cache_data(show_spinner=False)
def parse_csv_bytes(content: bytes, file_name: str) -> pd.DataFrame:
    """Money Forwardのエクスポート（UTF-8 BOM付き / CP932）をパースする。

    引数がバイト列とファイル名だけなので、同じ内容の再アップロードは
    st.cache_data によりキャッシュされ、リランごとの再パースを避けられる。
    """
    last_error: Exception | None = None
    frame: pd.DataFrame | None = None
    for encoding in ("utf-8-sig", "cp932"):
        try:
            frame = pd.read_csv(BytesIO(content), encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            # ParserErrorも捕捉して次のエンコーディングを試す
            # （誤ったエンコーディングで区切りが壊れるケースがあるため）
            last_error = error
    if frame is None:
        raise ValueError(f"CSVを読み込めません: {last_error}")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        names = "、".join(sorted(missing))
        raise ValueError(f"必要な列がありません: {names}")

    frame["元ファイル"] = file_name
    return frame


def read_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    return parse_csv_bytes(uploaded_file.getvalue(), uploaded_file.name)


def normalize_data(frames: list[pd.DataFrame]) -> pd.DataFrame:
    data = pd.concat(frames, ignore_index=True).copy()
    data["日付"] = pd.to_datetime(data["日付"], errors="coerce")
    data["金額（円）"] = pd.to_numeric(
        data["金額（円）"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    data = data.dropna(subset=["日付", "金額（円）"])
    data["計算対象"] = pd.to_numeric(data["計算対象"], errors="coerce").fillna(0)
    data["振替"] = pd.to_numeric(data["振替"], errors="coerce").fillna(0)
    data["大項目"] = data["大項目"].fillna("未分類")
    data["中項目"] = data["中項目"].fillna("未分類")

    # 期間が重複する複数ファイル（例: 月次と年次のエクスポート）による
    # 二重計上を防ぐ。Money ForwardのID列があれば明細単位で重複排除する。
    if "ID" in data.columns:
        has_id = data["ID"].notna() & (data["ID"].astype(str).str.strip() != "")
        with_id = data[has_id].drop_duplicates(subset="ID", keep="first")
        without_id = data[~has_id]
        data = pd.concat([with_id, without_id], ignore_index=True)

    return data.sort_values("日付", ascending=False)


def build_mock_data() -> pd.DataFrame:
    """デプロイ時のデモ表示用に6か月分の代表的なデータを生成する。"""
    categories = [
        ("食費", "食料品", 32_000, 1_200),
        ("住宅", "家賃・地代", 85_000, 0),
        ("水道・光熱費", "電気代", 12_000, 450),
        ("通信費", "インターネット", 5_500, 0),
        ("趣味・娯楽", "映画・音楽・ゲーム", 11_000, 800),
    ]
    # サーバーがUTC等の場合でも日本時間基準の「今月」になるようにする
    today_jst = pd.Timestamp.now(tz=JST).tz_localize(None).normalize()
    latest_month = today_jst.replace(day=1)
    months = pd.date_range(end=latest_month, periods=6, freq="MS")
    records = []
    for month_index, month in enumerate(months):
        for category_index, (major, minor, base_amount, monthly_change) in enumerate(categories):
            records.append(
                {
                    "計算対象": 1,
                    "日付": month + pd.Timedelta(days=3 + category_index * 4),
                    "内容": f"モック {minor}",
                    "金額（円）": -(base_amount + monthly_change * month_index),
                    "保有金融機関": "モック口座",
                    "大項目": major,
                    "中項目": minor,
                    "メモ": "デプロイ表示用のサンプルデータ",
                    "振替": 0,
                    "ID": f"mock-{month_index}-{category_index}",
                    "元ファイル": "モックデータ",
                }
            )
    return pd.DataFrame(records)


def select_transactions(
    data: pd.DataFrame,
    transaction_type: str,
    calculated_only: bool,
    exclude_transfers: bool,
) -> pd.DataFrame:
    selected = data.copy()
    if calculated_only:
        selected = selected[selected["計算対象"] == 1]
    if exclude_transfers:
        selected = selected[selected["振替"] == 0]

    amount = selected["金額（円）"]
    if transaction_type == "支出":
        selected = selected[amount < 0].copy()
        selected["集計金額"] = -selected["金額（円）"]
    elif transaction_type == "収入":
        selected = selected[amount > 0].copy()
        selected["集計金額"] = selected["金額（円）"]
    else:
        # フィルタ後のスライスへの代入によるSettingWithCopyWarning /
        # Copy-on-Write下での未反映を避けるため、収支でも明示的にコピーする
        selected = selected.copy()
        selected["集計金額"] = selected["金額（円）"]
    return selected


def cumulative_by_month(data: pd.DataFrame) -> pd.DataFrame:
    daily = data.groupby("日付", as_index=False)["集計金額"].sum()
    daily["月"] = daily["日付"].dt.strftime("%Y-%m")
    daily["日"] = daily["日付"].dt.day
    daily = daily.sort_values(["月", "日"])
    daily["月内累積額"] = daily.groupby("月")["集計金額"].cumsum()
    return daily


def build_cumulative_figure(
    cumulative: pd.DataFrame,
    transaction_type: str,
    y_axis_max: float | None,
    chart_template: str,
) -> go.Figure:
    figure = go.Figure()
    for month, values in cumulative.groupby("月", sort=True):
        figure.add_trace(
            go.Scatter(
                x=values["日"],
                y=values["月内累積額"],
                mode="lines+markers",
                name=month,
            )
        )
    figure.update_layout(
        template=chart_template,
        title=f"月ごとの累積{transaction_type}",
        xaxis_title="日",
        yaxis_title="金額（円）",
        legend_title="月",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    figure.update_xaxes(dtick=2, range=[1, 31])
    if y_axis_max is not None:
        # 収支モードでは累積額が負になり得るため、下限0で固定すると
        # データが見えなくなる。実データの最小値から下限を決める。
        minimum_value = float(cumulative["月内累積額"].min())
        lower_bound = min(0.0, minimum_value * 1.05)
        figure.update_yaxes(range=[lower_bound, y_axis_max])
    return figure


def monthly_category_totals(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    monthly = data.assign(月=data["日付"].dt.to_period("M"))
    return (
        monthly.groupby(["月", group_column], as_index=False)["集計金額"]
        .sum()
        .sort_values("月")
    )


def predict_monthly_spending(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """分類別の月額履歴から最新月の支出を推定する。

    scikit-learnの単回帰と等価な np.polyfit(1次) を使い、依存を軽くしている。
    学習点が少ない分類は外挿が不安定になるため、履歴の平均にフォールバックする。
    """
    monthly = monthly_category_totals(data, group_column)
    periods = sorted(monthly["月"].unique())
    if len(periods) < MINIMUM_HISTORY_MONTHS + 1:
        return pd.DataFrame()

    target_month = periods[-1]
    predictions = []
    for category, values in monthly.groupby(group_column):
        values = (
            values.set_index("月")["集計金額"]
            .reindex(periods, fill_value=0)
            .reset_index()
        )
        history = values.iloc[:-1]
        amounts = history["集計金額"].to_numpy(dtype=float)
        if len(amounts) >= MINIMUM_HISTORY_MONTHS:
            slope, intercept = np.polyfit(np.arange(len(amounts)), amounts, deg=1)
            predicted_amount = slope * len(amounts) + intercept
        else:
            predicted_amount = amounts.mean()
        predicted_amount = max(0.0, float(predicted_amount))
        actual_amount = values.loc[values["月"] == target_month, "集計金額"].iloc[0]
        predictions.append(
            {
                group_column: category,
                "予測月": str(target_month),
                "当月実績": actual_amount,
                "予測支出": predicted_amount,
            }
        )
    return pd.DataFrame(predictions).sort_values("予測支出", ascending=False)


def to_csv(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8-sig")


def resolve_chart_template(chart_theme: str) -> str:
    if chart_theme == "ダーク":
        return "plotly_dark"
    if chart_theme == "ライト":
        return "plotly_white"
    # 「自動」: st.context.theme はユーザーが実際に表示しているテーマを返す。
    # st.get_option("theme.base") はconfig.tomlの値しか返さないため使わない。
    try:
        theme_type = st.context.theme.type
    except AttributeError:
        # 古いStreamlitへのフォールバック
        theme_type = st.get_option("theme.base")
    return "plotly_dark" if theme_type == "dark" else "plotly_white"


AMOUNT_COLUMN_CONFIG = st.column_config.NumberColumn(format="localized")


st.set_page_config(page_title="家計簿ビューア", page_icon="bar_chart", layout="wide")
st.title("家計簿ビューア")
st.caption("Money ForwardのCSVをアップロードして、月別の累積額と分類別の内訳を確認します。")

if "uploaded_frames" not in st.session_state:
    st.session_state.uploaded_frames = {}
if "uploader_token" not in st.session_state:
    st.session_state.uploader_token = 0

uploaded_files = st.file_uploader(
    "CSVファイルをアップロード",
    type="csv",
    accept_multiple_files=True,
    key=f"csv_uploader_{st.session_state.uploader_token}",
)

for uploaded_file in uploaded_files:
    try:
        st.session_state.uploaded_frames[uploaded_file.name] = read_csv(uploaded_file)
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"{uploaded_file.name}: {error}")

data_source = st.radio(
    "表示データ",
    ["モックデータ", "アップロードしたCSV"],
    horizontal=True,
    index=0,
)

with st.expander("アップロード済みファイル", expanded=bool(st.session_state.uploaded_frames)):
    for name, frame in list(st.session_state.uploaded_frames.items()):
        left, middle, right = st.columns([4, 2, 1])
        left.write(name)
        middle.caption(f"{len(frame):,} 件")
        if right.button("削除", key=f"delete_{name}"):
            del st.session_state.uploaded_frames[name]
            st.session_state.uploader_token += 1
            st.rerun()

if data_source == "モックデータ":
    st.info("モックデータを表示しています。CSVをアップロード後に表示データを切り替えられます。")
    data = normalize_data([build_mock_data()])
elif not st.session_state.uploaded_frames:
    st.warning("アップロードしたCSVがありません。CSVを追加するか、モックデータを選択してください。")
    st.stop()
else:
    data = normalize_data(list(st.session_state.uploaded_frames.values()))

with st.sidebar:
    st.header("集計条件")
    transaction_type = st.selectbox("対象", ["支出", "収入", "収支"], index=0)
    group_column = st.radio("分類", ["大項目", "中項目"], horizontal=True)
    calculated_only = st.checkbox("計算対象のみ", value=True)
    exclude_transfers = st.checkbox("振替を除外", value=True)
    st.header("グラフ設定")
    chart_theme = st.selectbox("グラフテーマ", ["自動", "ライト", "ダーク"], index=0)
    maximum_y_value = st.number_input(
        "累積グラフの縦軸最大値（0で自動）",
        min_value=0,
        value=0,
        step=10_000,
    )

chart_template = resolve_chart_template(chart_theme)
y_axis_max = float(maximum_y_value) if maximum_y_value else None

selected = select_transactions(data, transaction_type, calculated_only, exclude_transfers)
if selected.empty:
    st.warning("現在の条件に一致する明細がありません。")
    st.stop()

total = selected["集計金額"].sum()
transaction_count = len(selected)
month_count = selected["日付"].dt.to_period("M").nunique()
metric_one, metric_two, metric_three = st.columns(3)
metric_one.metric(f"{transaction_type}合計", f"{total:,.0f} 円")
metric_two.metric("対象明細", f"{transaction_count:,} 件")
metric_three.metric("対象月数", f"{month_count} か月")

st.subheader("月ごとの累積額")
cumulative = cumulative_by_month(selected)
figure = build_cumulative_figure(cumulative, transaction_type, y_axis_max, chart_template)
st.plotly_chart(figure, width="stretch")

st.subheader(f"{group_column}別集計")
summary = (
    selected.groupby(group_column, as_index=False)["集計金額"]
    .sum()
    .sort_values("集計金額", ascending=False)
)
monthly_summary = (
    selected.assign(月=selected["日付"].dt.strftime("%Y-%m"))
    .pivot_table(index="月", columns=group_column, values="集計金額", aggfunc="sum", fill_value=0)
    .sort_index()
)
summary_left, summary_right = st.columns([3, 2])
with summary_left:
    st.dataframe(
        summary,
        column_config={"集計金額": AMOUNT_COLUMN_CONFIG},
        width="stretch",
        hide_index=True,
    )
with summary_right:
    category_chart = px.bar(
        summary,
        x=group_column,
        y="集計金額",
        template=chart_template,
        labels={"集計金額": "金額（円）"},
    )
    category_chart.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(category_chart, width="stretch")

st.download_button(
    "分類別集計をCSVでダウンロード",
    data=to_csv(summary),
    file_name=f"summary_by_{group_column}.csv",
    mime="text/csv",
)

st.subheader(f"{group_column}ごとの月別推移")
monthly_by_category = monthly_category_totals(selected, group_column)
category_options = summary[group_column].tolist()
displayed_categories = st.multiselect(
    "グラフに表示する項目",
    category_options,
    default=category_options[: min(5, len(category_options))],
)
if displayed_categories:
    trend_data = monthly_by_category[monthly_by_category[group_column].isin(displayed_categories)].copy()
    trend_data["月"] = trend_data["月"].astype(str)
    trend_chart = px.line(
        trend_data,
        x="月",
        y="集計金額",
        color=group_column,
        markers=True,
        template=chart_template,
        labels={"集計金額": "金額（円）"},
    )
    trend_chart.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(trend_chart, width="stretch")
else:
    st.info("表示する項目を1件以上選択してください。")

st.subheader("最新月の支出予測")
if transaction_type != "支出":
    st.info("支出を選択すると予測を表示します。")
else:
    forecast = predict_monthly_spending(selected, group_column)
    if forecast.empty:
        st.info(
            f"予測にはデータ上の最新月に加えて{MINIMUM_HISTORY_MONTHS}か月以上の"
            "支出履歴が必要です。"
        )
    else:
        actual_total, forecast_total = st.columns(2)
        actual_total.metric("最新月の実績合計", f"{forecast['当月実績'].sum():,.0f} 円")
        forecast_total.metric("予測支出合計", f"{forecast['予測支出'].sum():,.0f} 円")
        forecast_left, forecast_right = st.columns([3, 2])
        with forecast_left:
            st.dataframe(
                forecast,
                column_config={
                    "当月実績": AMOUNT_COLUMN_CONFIG,
                    "予測支出": AMOUNT_COLUMN_CONFIG,
                },
                width="stretch",
                hide_index=True,
            )
        with forecast_right:
            forecast_chart = px.bar(
                forecast,
                x=group_column,
                y="予測支出",
                template=chart_template,
                labels={"予測支出": "予測金額（円）"},
            )
            forecast_chart.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
            st.plotly_chart(forecast_chart, width="stretch")
        st.caption(
            "予測は分類別の月額履歴（最新月を除く）に線形回帰を適用した推定値です。"
            "「最新月」はデータ上の最も新しい月で、明細が月の途中までしかない場合、"
            "実績は月末時点より小さく表示されます。"
        )
st.download_button(
    "月別分類集計をCSVでダウンロード",
    data=to_csv(monthly_summary.reset_index()),
    file_name=f"monthly_summary_by_{group_column}.csv",
    mime="text/csv",
)

st.subheader("アップロード明細")
visible_columns = [column for column in DETAIL_COLUMNS if column in data.columns]
st.dataframe(data[visible_columns], width="stretch", hide_index=True)
st.download_button(
    "明細をCSVでダウンロード",
    data=to_csv(data[visible_columns]),
    file_name="uploaded_transactions.csv",
    mime="text/csv",
)
