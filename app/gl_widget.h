#pragma once

#include <QOpenGLWidget>

#include "path_tracer.h"

class GLWidget : public QOpenGLWidget {
public:
    GLWidget(QWidget *parent = 0) : QOpenGLWidget(parent) {}

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    rays::PathTracer *m_pathTracer;
};
