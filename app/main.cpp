#include <fstream>

#include <QApplication>

#include <parsers/obj_parser.h>
#include <path_tracer.h>

#include "gl_widget.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    std::ifstream sceneFile("../scenes/cornell-box/CornellBox-Original.obj");
    rays::ObjParser parser(sceneFile);
    parser.parse();

    rays::PathTracer pt;

    GLWidget *glWidget = new GLWidget();
    glWidget->setFixedSize(640, 360);
    glWidget->show();

    return app.exec();
}
