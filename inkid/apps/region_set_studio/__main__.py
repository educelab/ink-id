'''Launch the Region Set Studio application.'''

import sys
from PySide6.QtWidgets import QApplication
from inkid.apps.region_set_studio import MainWindow


class Application(QApplication):
    '''Wrap the Region Set Studio application.'''

    def __init__(self, argv):
        super().__init__(argv)
        self.window = MainWindow()


def main():
    '''Launch the Region Set Studio application and exit when finished.'''
    sys.exit(Application(sys.argv).exec())


if __name__ == '__main__':
    main()
