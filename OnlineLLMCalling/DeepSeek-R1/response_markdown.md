好的，我將為您規劃一個簡單但完整的問答系統解決方案。以下是詳細的目錄結構和文件實現：

```
myqa/
├── app.py               # 主程序入口
├── cli.py               # 命令行交互邏輯
├── config.py            # 配置文件
├── db/
│   ├── database.py      # 數據庫操作核心
│   └── init_db.py       # 數據庫初始化腳本
├── models/
│   └── qa_pair.py       # 數據模型定義
└── tests/
    └── test_basic.py    # 單元測試
```

以下是各文件的完整實現：

**config.py**
```python
import pathlib

# 基礎目錄路徑
BASE_DIR = pathlib.Path(__file__).parent.parent
DB_NAME = "qa_database.db"
DB_PATH = BASE_DIR / "db" / DB_NAME
SIMILARITY_THRESHOLD = 0.6  # 相似度閾值
```

**models/qa_pair.py**
```python
from datetime import datetime
from dataclasses import dataclass

@dataclass
class QAPair:
    question: str
    answer: str
    created_at: datetime = datetime.now()
```

**db/init_db.py**
```python
import sqlite3
from config import DB_PATH

def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT UNIQUE,
                answer TEXT,
                created_at TIMESTAMP
            )
        """)
        conn.commit()

if __name__ == "__main__":
    initialize_database()
```

**db/database.py**
```python
import sqlite3
from difflib import get_close_matches
from config import DB_PATH, SIMILARITY_THRESHOLD
from models.qa_pair import QAPair
from typing import Optional, List

class QADatabase:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row

    def _get_all_questions(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT question FROM qa_pairs")
        return [row["question"] for row in cursor.fetchall()]

    def find_best_match(self, question: str) -> Optional[str]:
        questions = self._get_all_questions()
        matches = get_close_matches(
            question.lower(), 
            [q.lower() for q in questions],
            n=1,
            cutoff=SIMILARITY_THRESHOLD
        )
        if matches:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT answer FROM qa_pairs WHERE LOWER(question) = ?",
                (matches[0],)
            )
            result = cursor.fetchone()
            return result["answer"] if result else None
        return None

    def insert_pair(self, pair: QAPair):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO qa_pairs (question, answer, created_at) VALUES (?, ?, ?)",
                (pair.question, pair.answer, pair.created_at)
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass  # 忽略重複問題

    def __del__(self):
        self.conn.close()
```

**cli.py**
```python
from db.database import QADatabase
from models.qa_pair import QAPair

class QACLI:
    def __init__(self):
        self.db = QADatabase()

    def _get_new_answer(self, question: str) -> str:
        print("沒有找到答案，請提供解答：")
        while True:
            answer = input("> ").strip()
            if answer:
                return answer
            print("答案不能為空，請重新輸入")

    def start(self):
        print("問答系統啟動 (輸入 'exit' 退出)")
        while True:
            question = input("\n問題：").strip()
            if question.lower() == 'exit':
                break
            
            answer = self.db.find_best_match(question)
            if answer:
                print(f"\n答案：{answer}")
            else:
                new_answer = self._get_new_answer(question)
                self.db.insert_pair(QAPair(
                    question=question,
                    answer=new_answer
                ))
                print("已儲存新解答！")
```

**app.py**
```python
from cli import QACLI

def main():
    cli = QACLI()
    cli.start()

if __name__ == "__main__":
    main()
```

**tests/test_basic.py**
```python
import unittest
from db.database import QADatabase
from models.qa_pair import QAPair
from datetime import datetime

class TestQASystem(unittest.TestCase):
    def setUp(self):
        self.db = QADatabase()
        self.test_pair = QAPair(
            question="如何重置密碼？",
            answer="訪問設置頁面的密碼重置選項",
            created_at=datetime.now()
        )

    def test_insert_and_retrieve(self):
        self.db.insert_pair(self.test_pair)
        answer = self.db.find_best_match("重置密碼的方法")
        self.assertIsNotNone(answer)
        self.assertIn("密碼重置", answer)

    def test_non_existing_question(self):
        answer = self.db.find_best_match("不存在的問題")
        self.assertIsNone(answer)

if __name__ == "__main__":
    unittest.main()
```

使用說明：
1. 先運行 `python db/init_db.py` 初始化數據庫
2. 主程序運行 `python app.py`
3. 測試運行 `python -m unittest tests/test_basic.py`

這個實現包含以下關鍵特性：
1. 使用SQLite進行數據存儲
2. 基於difflib的模糊匹配算法
3. 防止重複問題的數據完整性保護
4. 清晰的模塊化結構
5. 包含基本單元測試
6. 類型提示和文檔支持

您可以根據需要調整 SIMILARITY_THRESHOLD 參數來控制匹配靈敏度，或擴展數據庫模塊來添加更複雜的搜索算法（如TF-IDF或神經網絡模型）。