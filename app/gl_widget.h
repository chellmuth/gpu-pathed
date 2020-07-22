#pragma once

#include <QOpenGLWidget>

#include "render_session.h"

class GLWidget : public QOpenGLWidget {
public:
    GLWidget(QWidget *parent = 0) : QOpenGLWidget(parent) {}

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    rays::RenderSession *m_renderSession;
};
