#include "gl_widget.h"

#include <QOpenGLFunctions_4_5_Compatibility>

#include <render_session.h>
#include <hit_test.h>

void GLWidget::initializeGL()
{
    const int width = this->width();
    const int height = this->height();

    auto *f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_4_5_Compatibility>();
    f->glClearColor(1.f, 0.5f, 1.f, 1.f);

    GLuint pbos[2];
    for (int i = 0; i < 2; i++) {
        f->glGenBuffers(1, &pbos[i]);
        f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[i]);
        f->glBufferData(
            GL_PIXEL_UNPACK_BUFFER,
            4 * sizeof(GLubyte) * width * height,
            NULL,
            GL_DYNAMIC_DRAW
        );
        f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    m_renderSession = new rays::RenderSession(width, height);
    m_currentState = m_renderSession->init(pbos[0], pbos[1]);

    m_renderSession->hitTest(100, 100);
}

void GLWidget::resizeGL(int w, int h)
{}

void GLWidget::paintGL()
{
    if (m_currentState.isRendering) {
        m_currentState = m_renderSession->pollRender();
    } else {
        m_currentState = m_renderSession->renderAsync();
    }

    const int width = this->width();
    const int height = this->height();

    auto *f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_4_5_Compatibility>();

    f->glClear(GL_COLOR_BUFFER_BIT);

    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_currentState.pbo);

    f->glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    update();
}
