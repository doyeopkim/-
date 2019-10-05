# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mygui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import os
import cv2
import time
from time import sleep
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *
import tensorflow as tf
import pymysql
from threading import Thread
from object_detection.utils import label_map_util as lmu
from object_detection.utils import visualization_utils2 as vis_util

'''
객체인식 원형코드 사용시from object_detection.utils import visualization_utils as vis_util
객체인식 변형코드 사용시 from object_detection.utils import visualization_utils2 as vis_util <<<----- 번호판 인식모델
'''
from object_detection.utils.visualization_utils2 import car_info  # <<< utils2 를 설정했을때 사용할것
# from object_detection.utils import ops as utils_ops

conn = pymysql.connect(host='localhost', user='root', password='1234', db='Car_Num', charset='utf8')
# host = DB주소(localhost 또는 ip주소), user = DB id, password = DB password, db = DB명
curs = conn.cursor()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class Ui_Dialog(QWidget, object):

    def setupUi(self, Dialog):
        Dialog.resize(1280, 720)
        Dialog.setStatusTip("")
        Dialog.setStyleSheet("background-color: rgb(255, 255, 255);")
        Dialog.setWindowIcon(QtGui.QIcon('image/123.jpg'))  # WindowIcon 설정
        Dialog.setWindowTitle('Oil Shock - The Fuel Classifier System')
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(False)

        self.Main_lb = QtWidgets.QLabel(Dialog)
        self.Main_lb.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        pixmap = QPixmap('image/theme.jpg')
        pixmap = pixmap.scaled(1280, 720)  # 사이즈 재설정
        self.Main_lb.setPixmap(pixmap)

        self.frame = QtWidgets.QFrame(Dialog)  # 결과창 프레임
        self.frame.setGeometry(QtCore.QRect(650, 190, 580, 480))
        self.frame.setStyleSheet(
            "border-radius: 10px; background-color: rgb(204, 204, 204, 100);")  # 204 255 255 // 102 204 255
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)

        '''버튼&아이콘 생성자'''
        # 시작 버튼
        self.Rec_button = QtWidgets.QPushButton(Dialog)
        self.Rec_button.setGeometry(QtCore.QRect(1150, 30, 100, 100))
        self.Rec_button.setStyleSheet('background-color: rgb(240, 240, 240);')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap('../Numplate/image/play-button.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Rec_button.setIcon(icon)
        self.Rec_button.setIconSize(QtCore.QSize(50, 50))
        self.Rec_button.clicked.connect(self.Rec_button_clicked)  # 카메라 버튼이벤트 생성

        # 등록 버튼
        self.Register_button = QtWidgets.QPushButton(Dialog)
        self.Register_button.setGeometry(QtCore.QRect(950, 560, 140, 50))
        self.Register_button.setStyleSheet(
            'border-radius: 5px; background-color: rgb(051, 051, 051); color : rgb(255, 255, 255); font-size: 24pt; font-family: 맑은 고딕;')  # 255 102 051
        self.Register_button.setText('등록하기')
        self.Register_button.clicked.connect(self.Register_button_clicked)  # 취소 버튼이벤트 생성
        self.Register_button.setVisible(False)

        # 취소 버튼
        self.Cancel_button = QtWidgets.QPushButton(Dialog)
        self.Cancel_button.setGeometry(QtCore.QRect(790, 560, 140, 50))
        self.Cancel_button.setStyleSheet(
            'border-radius: 5px; background-color: rgb(051, 051, 051); color : rgb(255, 255, 255); font-size: 24pt; font-family: 맑은 고딕;')  # 255 102 051
        self.Cancel_button.setText('취 소')
        self.Cancel_button.clicked.connect(self.Cancel_button_clicked)  # 취소 버튼이벤트 생성
        self.Cancel_button.setVisible(False)

        # 확인 버튼
        self.Confirm_button = QtWidgets.QPushButton(Dialog)
        self.Confirm_button.setGeometry(QtCore.QRect(950, 560, 140, 50))
        self.Confirm_button.setStyleSheet(
            'border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 24pt; font-family: 맑은 고딕;')
        self.Confirm_button.setText('확 인')
        self.Confirm_button.clicked.connect(self.Confirm_button_clicked)  # 확인 버튼이벤트 생성
        self.Confirm_button.setVisible(False)

        '''라벨생성자'''
        # 로고 이미지 라벨
        self.Logo_lb = QtWidgets.QLabel(Dialog)
        self.Logo_lb.setGeometry(QtCore.QRect(50, 30, 350, 126))
        self.Logo_lb.setStyleSheet("background-color: rgb()")
        pixmap = QPixmap('image/logo.png')
        # pixmap = pixmap.scaled(350, 126)  # 사이즈 재설정
        self.Logo_lb.setPixmap(pixmap)

        # 디자인용 선 라벨
        '''self.Line_lb = QtWidgets.QLabel(Dialog)
        self.Line_lb.setGeometry(QtCore.QRect(170, 160, 1080, 4))
        self.Line_lb.setStyleSheet('background-color: rgb(000, 000, 000);')'''

        # 영상이 나올 라벨
        self.Video_lb = QtWidgets.QLabel(Dialog)
        self.Video_lb.setGeometry(QtCore.QRect(50, 190, 580, 480))
        self.Video_lb.setStyleSheet(
            'border : 4px solid black; border-radius: 10px; background-color: rgb(204, 204, 204, 100); font-size: 30pt; font-family: 맑은 고딕;')  # 폰트&사이즈
        self.Video_lb.setText('여기에 카메라 \n영상이 재생됩니다.')
        self.Video_lb.setAlignment(QtCore.Qt.AlignCenter)  # 중앙 정렬

        # 번호판 이미지 라벨
        self.Plate_img_lb = QtWidgets.QLabel(Dialog)
        self.Plate_img_lb.setGeometry(QtCore.QRect(810, 250, 260, 50))
        self.Plate_img_lb.setStyleSheet('background-color: rgb(000, 000, 000);')
        # pixmap = QPixmap('00.jpg')
        # pixmap = pixmap.scaled(140, 140)
        # self.Plate_img_lb.setPixmap(pixmap)
        self.Plate_img_lb.setVisible(False)

        # 번호판 라벨
        self.Num_Plate_lb = QtWidgets.QLabel(Dialog)
        self.Num_Plate_lb.setGeometry(QtCore.QRect(785, 315, 310, 60))  # 785 290 31 60
        self.Num_Plate_lb.setStyleSheet(
            'background-color: rgb(); font-weight : bold; font-size: 36pt; font-family: 맑은 고딕;')
        self.Num_Plate_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Num_Plate_lb.setText('')
        self.Num_Plate_lb.setVisible(False)

        # (고객님의 유종은)Text 라벨
        self.Cus_oil_lb = QtWidgets.QLabel(Dialog)
        self.Cus_oil_lb.setGeometry(QtCore.QRect(740, 380, 400, 60))
        self.Cus_oil_lb.setStyleSheet('background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Cus_oil_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Cus_oil_lb.setText('고객님 차량의 유종은')
        self.Cus_oil_lb.setVisible(False)

        # 유종 정보 라벨
        self.Oil_type_lb = QtWidgets.QLabel(Dialog)
        self.Oil_type_lb.setGeometry(QtCore.QRect(747, 430, 264, 60))
        self.Oil_type_lb.setStyleSheet(
            'color : rgb(000, 000, 000); background-color: rgb(); font-weight : bold; font-size: 30pt; font-family: 맑은 고딕;')
        self.Oil_type_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Oil_type_lb.setText('휘발유(가솔린)')
        self.Oil_type_lb.setVisible(False)

        # (입니다)Text 라벨
        self.Ex_last_lb = QtWidgets.QLabel(Dialog)
        self.Ex_last_lb.setGeometry(QtCore.QRect(1015, 430, 130, 60))
        self.Ex_last_lb.setStyleSheet('background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Ex_last_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Ex_last_lb.setText('입니다.')
        self.Ex_last_lb.setVisible(False)

        # (유종 정보가 맞다면 (확인)을 눌러주세요.)Text 라벨
        self.Plz_continue_lb = QtWidgets.QLabel(Dialog)
        self.Plz_continue_lb.setGeometry(QtCore.QRect(723, 510, 434, 30))
        self.Plz_continue_lb.setStyleSheet('background-color: rgb(); font-size: 18pt; font-family: 맑은 고딕;')
        self.Plz_continue_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Plz_continue_lb.setText('유종 정보가 맞다면         을 눌러주세요.')
        self.Plz_continue_lb.setVisible(False)

        # (확인)Text 라벨
        self.Check_lb = QtWidgets.QLabel(Dialog)
        self.Check_lb.setGeometry(QtCore.QRect(937, 510, 60, 30))
        self.Check_lb.setStyleSheet(
            'border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 18pt; font-family: 맑은 고딕;')
        self.Check_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Check_lb.setText('확 인')
        self.Check_lb.setVisible(False)

        # (유종 정보가 없다면 (등록하기)를 눌러주세요.)Text 라벨
        self.Plz_register_lb = QtWidgets.QLabel(Dialog)
        self.Plz_register_lb.setGeometry(QtCore.QRect(703, 510, 474, 30))
        self.Plz_register_lb.setStyleSheet('background-color: rgb(); font-size: 18pt; font-family: 맑은 고딕;')
        self.Plz_register_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Plz_register_lb.setText('유종 정보가 없다면              를 눌러주세요.')
        self.Plz_register_lb.setVisible(False)

        # (등록하기)Text 라벨
        self.Register_lb = QtWidgets.QLabel(Dialog)
        self.Register_lb.setGeometry(QtCore.QRect(918, 510, 100, 30))
        self.Register_lb.setStyleSheet(
            'border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 18pt; font-family: 맑은 고딕;')
        self.Register_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Register_lb.setText('등록하기')
        self.Register_lb.setVisible(False)

        # 프레임 라벨
        self.Fps_lb = QtWidgets.QLabel(Dialog)
        self.Fps_lb.setGeometry(QtCore.QRect(55, 643, 81, 16))
        self.Fps_lb.setStyleSheet('background-color: rgb(); color : white; font-size: 10pt; font-family: 맑은 고딕;')

        # 제작자 라벨
        self.Maker_lb = QtWidgets.QLabel(Dialog)
        self.Maker_lb.setGeometry(QtCore.QRect(990, 694, 274, 16))
        self.Maker_lb.setStyleSheet('background-color: rgb(); font-size: 10pt; font-family: 맑은 고딕;')
        self.Maker_lb.setAlignment(QtCore.Qt.AlignRight)
        self.Maker_lb.setText('The Fuel Classifier System  |  Team. Oil Shock')

    def setImage(self, image):  # 이미지를 라벨에 넣는 함수
        ui.Video_lb.setPixmap(QtGui.QPixmap.fromImage(image))

    # Event 함수
    def Rec_button_clicked(self):  # 시작 버튼 이벤트
        th1 = Thread(self)
        th1.changePixmap.connect(self.setImage)
        th2 = Thread2(self)

        th1.start()
        print('스레드1시작')
        th2.start()
        print('스레드2시작')

    def Register_button_clicked(self):  # 등록 버튼 이벤트
        print('등록')

    def Cancel_button_clicked(self):  # 취소 버튼 이벤트
        print('취소')

    def Confirm_button_clicked(self):  # 확인 버튼 이벤트
        print('확인')


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        prevtime = 0

        while True:
            ret, frame = capture.read()
            global re, fr
            re = ret
            fr = frame

            # 프레임 표시
            curtime = time.time()
            sec = curtime - prevtime
            prevtime = curtime
            fps = 1 / sec
            str = "FPS : %0.1f" % fps
            # cv2.putText(frame, str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            ui.Fps_lb.setText(str)
            # end 프레임

            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            sleep(0)


def Regi_ui():
    ui.Register_button.setVisible(True)
    ui.Cancel_button.setVisible(True)
    ui.Plate_img_lb.setVisible(True)
    ui.Num_Plate_lb.setVisible(True)
    ui.Plz_register_lb.setVisible(True)
    ui.Register_lb.setVisible(True)


def Exis_ui():
    ui.Confirm_button.setVisible(True)
    ui.Cancel_button.setVisible(True)
    ui.Plate_img_lb.setVisible(True)
    ui.Num_Plate_lb.setVisible(True)
    ui.Cus_oil_lb.setVisible(True)
    ui.Oil_type_lb.setVisible(True)
    ui.Ex_last_lb.setVisible(True)
    ui.Plz_continue_lb.setVisible(True)
    ui.Plz_register_lb.setVisible(True)
    ui.Register_lb.setVisible(True)


class Thread2(QThread):

    def run(self):
        time1 = time.time()
        MIN_ratio = 0.60

        # MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
        GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
        LABEL_FILE = 'data/mscoco_label_map.pbtxt'
        NUM_CLASSES = 90
        # end define

        label_map = lmu.load_labelmap(LABEL_FILE)
        categories = lmu.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        categories_index = lmu.create_category_index(categories)

        print("call label_map & categories : %0.5f" % (time.time() - time1))

        graph_file = MODEL_NAME + '/' + GRAPH_FILE_NAME

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sses = tf.Session(graph=detection_graph)

        print("store in memoey time : %0.5f" % (time.time() - time1))

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        print("road Video time : %0.5f" % (time.time() - time1))

        while True:
            global re, fr
            ret = re
            frame = fr
            frame_expanded = np.expand_dims(frame, axis=0)

            if not ret:
                print("나간다")
                break

            (boxes, scores, classes, nums) = sses.run(  # np.ndarray
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded}
            )  # end sses.run()

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                categories_index,
                use_normalized_coordinates=True,
                min_score_thresh=MIN_ratio,  # 최소 인식률
                line_thickness=2)  # 선두께

            try:
                pixmap = QPixmap('00.jpg')
                pixmap = pixmap.scaled(260, 50)

                ui.Num_Plate_lb.setText(str(car_info[0]))
                ui.Plate_img_lb.setPixmap(pixmap)

                curs = conn.cursor()

                sql = 'SELECT oil_type from NumPlate Where car_num = ' + "'" + str(car_info[0]) + "'"  # 실행 할 쿼리문 입력
                print(sql)
                curs.execute(sql)  # 쿼리문 실행
                rows = curs.fetchone()  # 데이터 패치
                print(rows)

                if rows == ('G',):
                    ui.Oil_type_lb.setText('휘발휘바')
                    Exis_ui()

                elif rows == ('D',):
                    ui.Oil_type_lb.setText("디제디제")
                    Exis_ui()
                else:
                    Regi_ui()
                conn.close()

            except:
                pass
            sleep(0)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    changePixmap = pyqtSignal(QImage)

    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()

    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture("asdf.mp4")  # 165145 162900

    sys.exit(app.exec_())
