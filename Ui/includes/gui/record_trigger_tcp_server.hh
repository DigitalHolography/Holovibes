/*! \file record_trigger_tcp_server.hh
 *
 * \brief Lightweight TCP notification server for external record triggers.
 */
#pragma once

#include <cstdint>

#include <QTcpServer>
#include <QTcpSocket>
#include <QVector>

namespace holovibes::gui
{

/*! \class RecordTriggerTcpServer
 *
 * \brief TCP server used to notify connected clients when a recording starts.
 */
class RecordTriggerTcpServer
{
  public:
    static RecordTriggerTcpServer& instance();

    void configure(bool enabled, quint16 port);
    void notify_record_started();

    bool enabled() const { return enabled_; }
    quint16 port() const { return port_; }

  private:
    RecordTriggerTcpServer();
    ~RecordTriggerTcpServer();

    void start();
    void stop();
    void accept_pending_connections();
    void remove_client(QTcpSocket* client);

    QTcpServer server_;
    QVector<QTcpSocket*> clients_;
    bool enabled_ = false;
    quint16 port_ = 50000;
};

} // namespace holovibes::gui
