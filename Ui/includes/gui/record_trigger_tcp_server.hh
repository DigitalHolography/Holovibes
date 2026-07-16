/*! \file record_trigger_tcp_server.hh
 *
 * \brief Lightweight TCP server for external record triggers.
 */
#pragma once

#include <cstdint>
#include <functional>

#include <QByteArray>
#include <QHash>
#include <QTcpServer>
#include <QTcpSocket>
#include <QVector>

namespace holovibes::gui
{

/*! \class RecordTriggerTcpServer
 *
 * \brief TCP server used to exchange recording start triggers with connected clients.
 */
class RecordTriggerTcpServer
{
  public:
    static RecordTriggerTcpServer& instance();

    void configure(bool enabled, quint16 port);
    void set_record_start_callback(std::function<void()> callback);
    void notify_record_started();

    bool enabled() const { return enabled_; }
    quint16 port() const { return port_; }

  private:
    RecordTriggerTcpServer();
    ~RecordTriggerTcpServer();

    void start();
    void stop();
    void accept_pending_connections();
    void process_client_data(QTcpSocket* client);
    void remove_client(QTcpSocket* client);

    QTcpServer server_;
    QVector<QTcpSocket*> clients_;
    QHash<QTcpSocket*, QByteArray> receive_buffers_;
    std::function<void()> record_start_callback_;
    bool enabled_ = false;
    quint16 port_ = 50000;
};

} // namespace holovibes::gui
