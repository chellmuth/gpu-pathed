#pragma once

#include <QMainWindow>
#include <QPushButton>

class MainWindow : public QMainWindow {
public:
    explicit MainWindow(QWidget *parent = 0);

private:
    QPushButton *m_button;
};
