#include "main_window.h"

#include <QHBoxLayout>

#include "gl_widget.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    auto *layout = new QHBoxLayout();

    auto *glWidget = new GLWidget(this);
    glWidget->setFixedSize(400, 400);
    layout->addWidget(glWidget);

    auto *window = new QWidget();
    window->setLayout(layout);
    setCentralWidget(window);

    setFixedSize(QSize(420, 420));
}
