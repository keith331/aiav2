from PySide2 import QtGui
from PySide2.QtWidgets import QHeaderView
from PySide2 import QtCore
from PySide2.QtCore import Qt

class LogModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(LogModel, self).__init__()
        self._data = data
                
    def headerData(self, section , orientation, role=Qt.DisplayRole):
        titles = ['Name', 'Result', 'Score', 'Datetime']

        if role == Qt.DisplayRole: # only change what DisplayRole returns
            if orientation == Qt.Horizontal:
                return titles[section]
            elif orientation == Qt.Vertical:
                return f'{section + 1}'

        return super().headerData(section, orientation, role) # must have this line

    def removeRows(self, parent=QtCore.QModelIndex()):
        self.beginRemoveRows()
        # do actual data remove
        self.endRemoveRows()

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value =  self._data[index.row()][index.column()]

            return value

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])
