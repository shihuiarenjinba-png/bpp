---
name: qa-pptx-generator
description: 同梱のテンプレートPPTXとPythonスクリプトを使用して、複数の4択問題とその解説スライドを1つのPPTXファイルにまとめて生成します。ユーザーが「クイズを作成して」「4択問題のスライドをまとめて作って」と依頼した際に使用します。
---

# QA PPTX Generator

このスキルは、同梱されたPowerPointテンプレートと専用の書き換えスクリプトを使用して、高品質な4択問題と解説スライドを生成します。複数のクイズを1つのファイルにまとめて出力することが可能です。

## スキルに含まれるリソース
-   **templates/template.pptx**: デザイン済みのQ&Aスライドテンプレート（2枚構成）。
-   **scripts/generate_qa_pptx.py**: テンプレートを元に、複数クイズを1つのPPTXに結合・書き換えを行うPythonスクリプト。

## ワークフロー

1.  **コンテンツの生成**:
    ユーザーの要望（トピックや問題数など）に基づき、複数のQ&Aデータを含むリストを生成します。
    -   **各クイズの構成**:
        -   **問題スライド**: カテゴリ、問題文、4つの選択肢。
        -   **解説スライド**: 正解の選択肢、各選択肢の正誤理由を含む詳細な解説。

2.  **JSONデータの作成**:
    生成したコンテンツをリスト形式で一時的なJSONファイル（`/home/ubuntu/qa_data.json`）に保存します。
    ```json
    [
      {
        "question": {
          "category": "歴史",
          "text": "鎌倉幕府をひらいたのは誰？",
          "choices": ["源頼朝", "足利尊氏", "徳川家康", "織田信長"]
        },
        "answer": {
          "correct_choice": "源頼朝",
          "explanation": "源頼朝は1192年に征夷大将軍となり、鎌倉幕府を開きました。"
        }
      },
      {
        "question": {
          "category": "科学",
          "text": "水の化学式は？",
          "choices": ["CO2", "H2O", "O2", "N2"]
        },
        "answer": {
          "correct_choice": "H2O",
          "explanation": "水は水素原子2つと酸素原子1つからなる化合物です。"
        }
      }
    ]
    ```

3.  **スライドの生成**:
    スキル内のスクリプトを呼び出し、全クイズを含む1つのPPTXを生成します。
    ```bash
    python3 /home/ubuntu/skills/qa-pptx-generator/scripts/generate_qa_pptx.py /home/ubuntu/qa_data.json /home/ubuntu/output_qa.pptx
    ```

4.  **結果の提供**:
    生成された `/home/ubuntu/output_qa.pptx` をユーザーに提供します。

## 注意事項
-   書き換えスクリプトは `python-pptx` ライブラリを使用します。環境にない場合は以下でインストールしてください。
    ```bash
    pip install python-pptx --break-system-packages -q
    ```
-   複数のクイズが指定された場合、各クイズは「問題スライド」と「解説スライド」の2枚1組で追加されます。
-   解説文は、各選択肢の違いが明確になるように具体的に記述してください。
