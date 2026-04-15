import sys
import re
import sqlite3
import pandas as pd
import PyPDF2

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()


# ---------------- UTIL ----------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text.lower())


def extract_text(path):
    text = ""
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return clean_text(text)


def show_message(title, msg):
    m = QMessageBox()
    m.setWindowTitle(title)
    m.setText(msg)
    m.setStyleSheet("""
        QMessageBox { background:#2b2b2b; color:white; }
        QLabel { color:white; }
        QPushButton { background:#444; color:white; padding:6px; }
    """)
    m.exec_()


# ---------------- LOGIN ----------------
class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        self.setFixedSize(420, 360)

        self.setStyleSheet("""
            QWidget { background:#121212; color:#e0e0e0; }
            QLineEdit {
                background:#1e1e1e;
                border:1px solid #333;
                border-radius:8px;
                padding:10px;
            }
            QLineEdit:focus { border:1px solid #3a7afe; }
            QPushButton {
                background:#2a2a2a;
                border-radius:8px;
                padding:10px;
            }
            QPushButton:hover { background:#3a3a3a; }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.addStretch()

        title = QLabel("Login")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))

        self.user = QLineEdit()
        self.user.setPlaceholderText("Username")

        self.passw = QLineEdit()
        self.passw.setPlaceholderText("Password")
        self.passw.setEchoMode(QLineEdit.Password)

        btn_login = QPushButton("Login")
        btn_reg = QPushButton("Register")

        btn_login.clicked.connect(self.login)
        btn_reg.clicked.connect(self.open_reg)

        layout.addWidget(title)
        layout.addWidget(self.user)
        layout.addWidget(self.passw)
        layout.addWidget(btn_login)
        layout.addWidget(btn_reg)
        layout.addStretch()

        self.setLayout(layout)

    def login(self):
        u = self.user.text().strip()
        p = self.passw.text().strip()

        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        if cursor.fetchone():
            self.main = ResumeApp()
            self.main.show()
            self.close()
        else:
            show_message("Error", "Invalid credentials")

    def open_reg(self):
        self.reg = RegisterWindow()
        self.reg.show()


# ---------------- REGISTER ----------------
class RegisterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Register")
        self.setFixedSize(320, 240)
        self.setStyleSheet("background:#121212; color:white;")

        layout = QVBoxLayout()

        title = QLabel("Register")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))

        self.user = QLineEdit()
        self.user.setPlaceholderText("Username")

        self.passw = QLineEdit()
        self.passw.setPlaceholderText("Password")
        self.passw.setEchoMode(QLineEdit.Password)

        btn = QPushButton("Create Account")
        btn.clicked.connect(self.register)

        layout.addWidget(title)
        layout.addWidget(self.user)
        layout.addWidget(self.passw)
        layout.addWidget(btn)

        self.setLayout(layout)

    def register(self):
        try:
            cursor.execute("INSERT INTO users VALUES (?,?)",
                           (self.user.text(), self.passw.text()))
            conn.commit()
            show_message("Success", "Account created")
            self.close()
        except:
            show_message("Error", "User already exists")


# ---------------- MAIN APP ----------------
class ResumeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resume Screening System")
        self.resize(1000, 650)

        self.setStyleSheet("""
            QWidget { background:#111; color:white; }
            QTextEdit {
                background:#1E1E1E;
                padding:10px;
                border-radius:8px;
            }
            QPushButton {
                background:#2A2A2A;
                padding:10px;
                border-radius:8px;
            }
            QPushButton:hover { background:#3A3A3A; }
            QTableWidget { background:#1E1E1E; }
            QHeaderView::section {
                background:#2c2c2c;
                color:white;
                padding:6px;
                border:1px solid #444;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Resume Screening System")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))

        self.job = QTextEdit()
        self.job.setPlaceholderText("Enter job description...")
        self.job.setFixedHeight(120)

        btn_layout = QHBoxLayout()

        btn1 = QPushButton("Upload Resumes")
        btn2 = QPushButton("Train Model")
        btn3 = QPushButton("Analyze")

        btn1.clicked.connect(self.upload)
        btn2.clicked.connect(self.train)
        btn3.clicked.connect(self.analyze)

        btn_layout.addWidget(btn1)
        btn_layout.addWidget(btn2)
        btn_layout.addWidget(btn3)

        self.table = QTableWidget()
        self.table.setMinimumHeight(350)

        main_layout.addWidget(title)
        main_layout.addWidget(self.job)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.table)

        self.setLayout(main_layout)

        self.vectorizer = TfidfVectorizer()
        self.model = SVC(kernel='linear', probability=True)
        self.files = []

    def upload(self):
        self.files, _ = QFileDialog.getOpenFileNames(self, "", "", "*.pdf")
        if self.files:
            show_message("Upload Complete", f"{len(self.files)} resumes loaded")

    def train(self):
        try:
            data = pd.read_csv("dataset.csv")

            texts = data["text"].apply(clean_text).tolist()
            labels = data["label"].tolist()

            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)

            show_message("Success", f"Model trained on {len(texts)} samples")

        except Exception as e:
            show_message("Error", f"Dataset error: {str(e)}")

    def analyze(self):
        job = clean_text(self.job.toPlainText())

        if not job or not self.files:
            show_message("Error", "Enter job description and upload resumes")
            return

        results = []

        for f in self.files:
            text = extract_text(f)

            vec = self.vectorizer.transform([text])

            sim = cosine_similarity(
                self.vectorizer.transform([job]), vec
            )[0][0]

            pred = self.model.predict(vec)[0]
            prob = self.model.predict_proba(vec)[0][1]

            level = "Strong" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"

            results.append((f.split("/")[-1], sim, pred, prob, level))

        results.sort(key=lambda x: x[1], reverse=True)

        self.table.setRowCount(len(results))
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Resume", "Similarity", "Prediction", "Confidence", "Match"]
        )

        for i, (name, sim, pred, prob, level) in enumerate(results):

            items = [
                QTableWidgetItem(name),
                QTableWidgetItem(f"{sim:.2f}"),
                QTableWidgetItem("Selected" if pred else "Rejected"),
                QTableWidgetItem(f"{prob:.2f}"),
                QTableWidgetItem(level)
            ]

            for item in items[1:]:
                item.setTextAlignment(Qt.AlignCenter)

            if i == 0:
                color = QColor("#1f3b5c")
                font = QFont()
                font.setBold(True)
                for item in items:
                    item.setBackground(color)
                    item.setFont(font)

            for j, item in enumerate(items):
                self.table.setItem(i, j, item)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    login = LoginWindow()
    login.show()
    sys.exit(app.exec_())