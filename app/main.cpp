#include <iostream>

#include <QApplication>

#include <path_tracer.h>

#include "main_window.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    std::cout << "Hello, world!" << std::endl;
    rays::PathTracer pt;

    MainWindow window;
    window.show();

    return app.exec();
}
