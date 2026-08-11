# 家計簿ビューア

🔗 **アプリを開く: https://household-budget-dashboard.streamlit.app/**

Money ForwardのエクスポートCSVをアップロードして、月別の累積額・分類別の内訳・支出予測を確認できるStreamlitアプリです。

## 使い方

1. Money Forwardから家計簿CSVをエクスポートする
2. アプリの「CSVファイルをアップロード」からCSVを追加する（複数可）
3. 「表示データ」を「アップロードしたCSV」に切り替える

CSVをアップロードしなくても、モックデータで動作イメージを確認できます。

## データの扱いについて

- アップロードされたCSVはブラウザのセッション中のみメモリ上で処理され、サーバーに永続保存されません。
- CSVのアップロードは自己責任でお願いします。気になる場合はローカルで実行してください。

## ローカルでの実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## デプロイ

Streamlit Community Cloudで `app.py` をメインファイルに指定してデプロイします。
Secretsの設定は不要です。
