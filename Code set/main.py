import sys

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import Qt

from FacialEmotionDetection import EmotionDetector
from loginUI import Ui_Form
from ui_main import Ui_MainWindow
import csv
import matplotlib.pyplot as plt


def main():
    global username
    app = QApplication(sys.argv)

    # login
    loginWindow = QMainWindow()
    loginUi = Ui_Form()
    loginUi.setupUi(loginWindow)
    loginWindow.show()

    # main window
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)


    ui.stackedWidget.setCurrentWidget(ui.page_home)
    ui.emoD_btn.clicked.connect(lambda: ui.stackedWidget.setCurrentWidget(ui.page_emod))
    ui.report_btn.clicked.connect(lambda: update_page_and_chart(username, ui))
    ui.da_btn.clicked.connect(lambda: ui.stackedWidget.setCurrentWidget(ui.page_da))
    ui.menuBtn.clicked.connect(lambda: ui.stackedWidget.setCurrentWidget(ui.page_home))
    ui.selectphoto.clicked.connect(lambda: select_photo(ui))


    def on_login_button_clicked():
        global username
        username = loginUi.lineEdit.text()
        password = loginUi.lineEdit_2.text()
        # print(username, password)
        loginWindow.close()
        mainWindow.show()

    loginUi.pushButton.clicked.connect(on_login_button_clicked)

    app.exec_()


def select_photo(ui):
    global username
    print("Select Photo button clicked")
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
    if file_dialog.exec_():
        selected_files = file_dialog.selectedFiles()
        if len(selected_files) > 0:
            file_path = selected_files[0]

            # print("Selected file path:", file_path)
            ed = EmotionDetector('emotion_detection_model_100epochs.h5', file_path)
            predicted_label = ed.predict()

            ui.pred_result.setText(predicted_label)
            image = QImage(file_path)
            pixmap = QPixmap.fromImage(image)
            ui.photo.setPixmap(pixmap)
            ui.photo.setScaledContents(True)

            with open('results.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([username, predicted_label])


def update_page_and_chart(username, ui):
    ui.stackedWidget.setCurrentWidget(ui.page_report)
    draw_pie_chart(username, ui)


def draw_pie_chart(username, ui):
    labels = []
    counts = []

    with open('results.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2 and row[0] == username:
                label = row[1]
                if label in labels:
                    index = labels.index(label)
                    counts[index] += 1
                else:
                    labels.append(label)
                    counts.append(1)
    print(1)

    if len(labels) > 0:
        fig, ax = plt.subplots()
        ax.pie(counts, labels=labels, autopct='%1.1f%%')
        ax.set_title(f"Emotion Distribution for {username}")
        ax.axis('equal')

        # save
        plt.savefig('pie_chart.png')

        # label
        pixmap = QPixmap('pie_chart.png')
        ui.label_chart.setPixmap(pixmap)
        ui.label_chart.setScaledContents(True)

        plt.close(fig)
    else:
        print(f"No data found for {username}.")


if __name__ == "__main__":
    main()
