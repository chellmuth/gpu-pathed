#include "gl_widget.h"

#include <iostream>

#include <QOpenGLFunctions_4_5_Compatibility>

GLuint pbo;

void GLWidget::initializeGL()
{
    const int width = this->width();
    const int height = this->height();

    auto *f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_4_5_Compatibility>();
    f->glClearColor(1.f, 0.5f, 1.f, 1.f);

	f->glGenBuffers(1, &pbo);
	f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	f->glBufferData(
        GL_PIXEL_UNPACK_BUFFER,
        4 * sizeof(GLubyte) * width * height,
        NULL,
        GL_DYNAMIC_DRAW
    );
	f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    m_pathTracer = new rays::PathTracer();
    m_pathTracer->init(pbo, width, height);
}

void GLWidget::resizeGL(int w, int h)
{}

void GLWidget::paintGL()
{
    const int width = this->width();
    const int height = this->height();

    auto *f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_4_5_Compatibility>();

    f->glClear(GL_COLOR_BUFFER_BIT);

    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    m_pathTracer->render();

    f->glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    update();
}
