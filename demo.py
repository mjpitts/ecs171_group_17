from cProfile import label
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QFont
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
import sys




def make_pred(data):
    #input vector = age	anaemia	creatinine_phosphokinase	diabetes	ejection_fraction	high_blood_pressure	platelets	serum_creatinine	serum_sodium  sex	smoking


    # actual prediction model
    df = pd.read_csv('heart_failure.csv')

    def label_high(row):
        if row['time'] <= 140 and row["DEATH_EVENT"] == 1:
            return 1
        elif row['time'] > 140 and row['DEATH_EVENT'] == 1:
            return -1
        elif row['time'] <= 140 and row['DEATH_EVENT'] == 0:
            return -1
        elif row['time'] > 140 and row['DEATH_EVENT'] == 0:
            return 0


    df['high'] =  df.apply(lambda row: label_high(row), axis=1)

    df_drop = df[(df.high == -1)]
    df.drop(index=df_drop.index, inplace=True,axis=1)
    df.shape

    df = df.drop(['DEATH_EVENT'], axis=1)

    # split dataframe into predictor set X and target y
    X_svm = df.drop(["high", "time"], axis=1)
    y_svm = df["high"]

    # split into training and test sets
    X_svm_train, X_svm_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size = 0.20, random_state=8)

    # Scale data
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(X_svm_train)
    X_train_std = sc.transform(X_svm_train)
    X_test_std = sc.transform(X_svm_test)

    #fit model
    svc_rbf = svm.SVC(kernel = 'rbf')
    svc_rbf.fit(X_train_std, y_train)

    #model is ready for predicitions
    return(svc_rbf.predict(data)[0])

def main():

    test = []
    test1 = []
    test2 = []
    test3 = []
    test4 = []
    test5 = []
    test6 = []
    test7 = []
    test8 = []
    test9 = []
    test10 = []


    print("Enter patient's full name, first last:")
    name = input()
    print("Enter patient's age:")
    test.append(float(input()))
    print("Does the patient have animia? Enter 1 for yes 0 for no:")
    test1.append(float(input()))
    print("Please enter patient's creatineine phosphokinase levels:")
    test2.append(float(input()))
    print("Does the patient have diabetes? Enter 1 for yes 0 for no:")
    test3.append(float(input())) 
    print("Please enter patient's ejection fraction:")
    test4.append(float(input()))
    print("Does the have high blood pressure? Enter 1 for yes 0 for no:")
    test5.append(float(input()))
    print("Please enter patient's platelet levels:")
    test6.append(float(input()))
    print("Please enter patient's serum creatineine levels:")
    test7.append(float(input()))
    print("Please enter patient's serum sodium levels:")
    test8.append(float(input()))
    print("Enter the sex of the patient, 1 for male, 0 for female")
    test9.append(float(input())) 
    print("Does the patient smoke? Enter 1 for yes 0 for no:")
    test10.append(float(input()))
    # df = pd.DataFrame( list(zip(test,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11)),
    #                     columns=['age', 'anaemia',	'creatinine_phosphokinase',	'diabetes',	'ejection_fraction',	'high_blood_pressure',	'platelets',	'serum_creatinine',	'serum_sodium',	'sex',	'smoking'])

    df = pd.DataFrame( list(zip(test,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10)))
    
    pred = make_pred(df)

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(0,0, 634, 817)
    win.setWindowTitle("Heart Failure Risk Prediction")

    font = QtGui.QFont()
    font.setFamily("Perpetua Titling MT")

    title = QtWidgets.QLabel(win)
    title.setText(f"Heart Failure Risk Predictor")
    title.setGeometry(100, 20, 421, 31)
    title.setFont(QFont('Arial', 23))

    name_label = QtWidgets.QLabel(win)
    name_label.setText(f"Patient's Name: {name}")
    name_label.setGeometry(100, 90, 400, 50)
    name_label.setFont(QFont('Arial', 20))
    
    label = QtWidgets.QLabel(win)
    label.setGeometry(QtCore.QRect(100, 20, 421, 31))
    label.setFont(font)
    label.setObjectName("label")
    label_2 = QtWidgets.QLabel(win)
    label_2.setGeometry(QtCore.QRect(40, 260, 147, 13))
    label_2.setObjectName("label_2")
    label_3 = QtWidgets.QLabel(win)
    label_3.setGeometry(QtCore.QRect(40, 350, 191, 16))
    label_3.setObjectName("label_3")
    label_4 = QtWidgets.QLabel(win)
    label_4.setGeometry(QtCore.QRect(40, 290, 181, 16))
    label_4.setObjectName("label_4")
    label_5 = QtWidgets.QLabel(win)
    label_5.setGeometry(QtCore.QRect(340, 290, 121, 16))
    label_5.setObjectName("label_5")
    label_6 = QtWidgets.QLabel(win)
    label_6.setGeometry(QtCore.QRect(340, 260, 147, 13))
    label_6.setObjectName("label_6")
    label_7 = QtWidgets.QLabel(win)
    label_7.setGeometry(QtCore.QRect(340, 320, 151, 16))
    label_7.setObjectName("label_7")
    label_8 = QtWidgets.QLabel(win)
    label_8.setGeometry(QtCore.QRect(440, 370, 147, 13))
    label_8.setObjectName("label_8")
    label_9 = QtWidgets.QLabel(win)
    label_9.setGeometry(QtCore.QRect(40, 320, 147, 13))
    label_9.setObjectName("label_9")
    label_miss = QtWidgets.QLabel(win)
    label_miss.setGeometry(QtCore.QRect(40, 380, 147, 13))
    label_miss.setObjectName("label_miss")
    label_11 = QtWidgets.QLabel(win)
    label_11.setGeometry(QtCore.QRect(40, 410, 151, 16))
    label_11.setObjectName("label_11")
    label_12 = QtWidgets.QLabel(win)
    label_12.setGeometry(QtCore.QRect(340, 350, 147, 13))
    label_12.setObjectName("label_12")
    label_13 = QtWidgets.QLabel(win)
    label_13.setGeometry(QtCore.QRect(340, 380, 171, 16))
    label_13.setObjectName("label_13")
    label_14 = QtWidgets.QLabel(win)
    label_14.setGeometry(QtCore.QRect(270, 130, 147, 21))
    label_14.setObjectName("label_14")
    label_15 = QtWidgets.QLabel(win)
    label_15.setGeometry(QtCore.QRect(360, 290, 147, 20))
    label_15.setObjectName("label_15")
    label_16 = QtWidgets.QLabel(win)
    label_16.setGeometry(QtCore.QRect(230, 90, 191, 20))
    label_16.setObjectName("label_16")
    label_17 = QtWidgets.QLabel(win)
    label_17.setGeometry(QtCore.QRect(200, 200, 211, 16))

    label_2.setText(f"Age: {test[0]}")

    label_3.setText(f"Creatinine Phosphokinase: {test2[0]}")

    if test5[0] == 1 : bp = "True" 
    else: bp = "False"
    label_4.setText(f"High blood pressure: {bp}")

    label_5.setText(f"Serum Sodium: {test8[0]}")

    label_6.setText(f"Platelets: {test6[0]}")

    label_7.setText(f"Serum Creatinine: {test7[0]}")

    if test1[0] == 1 : anemia = "True" 
    else: anemia = "False"
    label_9.setText(f"Anemia: {anemia}")

    if test3[0] == 1 : d = "True" 
    else: d = "False"
    label_miss.setText(f"Diabetes: {d}")

    label_11.setText(f"Ejection Fraction: {test4[0]}")

    if test10[0] == 1 : s = "True" 
    else: s = "False"
    label_12.setText(f"Smoking: {s}")
    
    if test9[0] == 1 : sex = "Male" 
    else: sex = "Female"
    label_13.setText(f"Male or Female: {sex}")

    resul_box = QtWidgets.QTextBrowser(win)
    resul_box.setGeometry(QtCore.QRect(130, 520, 331, 221))
    resul_box.setObjectName("resul_box")
    if pred == 1:
        resul_box.setHtml( "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:22pt;\">Patient\'s Risk Status:</span></p>\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; font-weight:600; color:#ff0000\">High Risk</span></p></body></html>")
    else:
        resul_box.setHtml( "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:22pt;\">Patient\'s Risk Status:</span></p>\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; font-weight:600; color:#00ff00\">Low Risk</span></p></body></html>")



    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    
