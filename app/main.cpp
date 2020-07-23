#include <fstream>

#include <QApplication>

#include "gl_widget.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    GLWidget *glWidget = new GLWidget();
    glWidget->setFixedSize(640, 360);
    glWidget->show();

    return app.exec();
}
