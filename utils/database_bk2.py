from PySide2.QtSql import *
from PySide2.QtCore import Qt

class TableModel(QSqlTableModel):

    def __init__(self):
        super().__init__()

        self.db = QSqlDatabase.addDatabase('QSQLITE')
        self.db.setDatabaseName('database.db')
        if not self.db.open():
            self.close()
        self.table_model = QSqlTableModel()
        self.table_model.setTable("Testlog")
        self.table_model.select()
        self.table_model.setHeaderData(0, Qt.Horizontal, "Name")
        self.table_model.setHeaderData(1, Qt.Horizontal, "Result")
        self.table_model.setHeaderData(2, Qt.Horizontal, "Score")

    def add_data(self, a, b, c):
        record = self.table_model.record()
        record.setValue('Name', a)
        record.setValue('Result',b)
        record.setValue('Score',c)
        self.table_model.insertRecord(self.table_model.rowCount(), record)