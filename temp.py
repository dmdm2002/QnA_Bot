import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from datetime import datetime
form_class = uic.loadUiType("pyqt_ui/template.ui")[0]

import os
from munch import Munch
from rag_main import get_configs, handling_database, setting_rag_system, chat_ai
from fastspeech2 import tts_inference


class ChatLog:
    def __init__(self):
        super().__init__()
        self.root = './system_log'
        os.makedirs(self.root, exist_ok=True)

    def save_log(self, speaker, question):
        with open(f'{self.root}/chat_log.txt', 'a') as f:
            now = datetime.now()
            current = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{current}] {speaker}: {question}\n')

    def load_log(self):
        log_path = f'{self.root}/chat_log.txt'
        if not os.path.exists(log_path):
            return ""
        with open(log_path, 'r') as f:
            return f.read()


class RagMain:
    def __init__(self):
        super().__init__()
        try:
            cfg = get_configs('config.yml')
            self.rag_cfg = Munch(cfg['rag'])
            print(f'-----Get Config Information-----')
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    def database_update(self):
        handling_database(self.rag_cfg)

    def get_chain(self):
        chain = setting_rag_system(self.rag_cfg)

        return chain


class TTSWorker(QThread):
    """스레드에서 TTS 작업을 처리하는 클래스"""
    finished = pyqtSignal()  # 작업 완료 시 시그널 발생

    def __init__(self, answer):
        super().__init__()
        self.answer = answer

    def run(self):
        # TTS 작업 실행
        tts_inference.inference(self.answer)
        # 작업 완료 시 신호 보내기
        self.finished.emit()


class MyWindow(QMainWindow, form_class):
    def __init__(self, chain):
        super().__init__()
        self.setupUi(self)
        self.chain = chain
        self.chat_history = []

        # ChatLog instance
        self.chat_log = ChatLog()

        # Connect button to event
        self.user_question_text_box_button.clicked.connect(self.user_question_text_clicked)

        # Enable Enter key for sending messages
        self.user_question_text_box.keyPressEvent = self.text_box_key_press
        self.chat_history = []  # 대화 기록을 수집

    def user_question_text_clicked(self):
        # Get user input
        question = self.user_question_text_box.toPlainText()

        # Save the log with the user's input
        self.chat_log.save_log("User", question)

        # Clear the input box
        self.user_question_text_box.clear()

        # Load the chat log and display it in the chat log box
        chat_log_text = self.chat_log.load_log()
        self.chat_log_box.setText(chat_log_text)

        # Get the AI answer and update chat history
        answer, self.chat_history = chat_ai(self.chain, question, chat_history=self.chat_history)

        # TTS 진행 상태 표시
        self.tts_status_label.setText("음성 파일 생성 진행 중...")

        # Run TTS in a separate thread
        self.tts_worker = TTSWorker(answer)
        self.tts_worker.finished.connect(self.on_tts_finished)
        self.tts_worker.start()

        # Save the log with the AI's output
        self.chat_log.save_log("AI", answer)

        # Immediately update the chat log with the answer
        chat_log_text = self.chat_log.load_log()
        self.chat_log_box.setText(chat_log_text)

        # ** Scroll to the bottom of the chat_log_box after updating **
        self.chat_log_box.moveCursor(QTextCursor.End)
        self.chat_log_box.ensureCursorVisible()

    def on_tts_finished(self):
        """TTS 작업 완료 시 호출"""
        self.tts_status_label.setText("음성 파일 생성 완료!")

    def text_box_key_press(self, event):
        if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier:
            # Enter key without Shift sends the message
            self.user_question_text_clicked()
            event.accept()
        elif event.key() == Qt.Key_Return and event.modifiers() & Qt.ShiftModifier:
            # Shift + Enter adds a new line
            cursor = self.user_question_text_box.textCursor()
            cursor.insertText("\n")
            self.user_question_text_box.setTextCursor(cursor)
            event.accept()
        else:
            # Default behavior for other keys
            QTextEdit.keyPressEvent(self.user_question_text_box, event)


if __name__ == "__main__":
    try:
        cfg = get_configs('config.yml')
        rag_cfg = Munch(cfg['rag'])
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    try:
        handling_database(rag_cfg)
    except Exception as e:
        print(f"Error updating database: {e}")
        sys.exit(1)

    try:
        chain = setting_rag_system(rag_cfg)
        if not chain:
            raise ValueError("Failed to initialize chain.")
    except Exception as e:
        print(f"Error setting up chain: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    myWindow = MyWindow(chain)
    myWindow.show()
    app.exec_()
