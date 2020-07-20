#include <iostream>

#include <QApplication>

#include <path_tracer.h>

#include "gl_widget.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    rays::PathTracer pt;

    GLWidget *glWidget = new GLWidget();
    glWidget->setFixedSize(640, 360);
    glWidget->show();

    return app.exec();
}
