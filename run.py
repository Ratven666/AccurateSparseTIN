from PyQt6.QtWidgets import QApplication

from interface.interface import ASTinUI

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = ASTinUI()
    ui.show()
    sys.exit(app.exec())
