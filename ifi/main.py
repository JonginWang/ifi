import tkinter as tk
import sys
import os

# 프로젝트의 루트 디렉토리를 Python 경로에 추가
# 이 스크립트가 'tek_automator' 폴더 내에 있다고 가정합니다.
# 만약 'ifi' 폴더에서 'python tek_automator/main.py'를 실행한다면,
# 아래 코드는 잘 동작할 것입니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .gui.main_window import Application

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

if __name__ == '__main__':
    main() 