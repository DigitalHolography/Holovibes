#include "record_trigger_tcp_server.hh"

#include <QHostAddress>

#include "logger.hh"

namespace holovibes::gui
{

RecordTriggerTcpServer& RecordTriggerTcpServer::instance()
{
    static RecordTriggerTcpServer server;
    return server;
}

RecordTriggerTcpServer::RecordTriggerTcpServer()
{
    QObject::connect(&server_, &QTcpServer::newConnection, &server_, [this]() { accept_pending_connections(); });
}

RecordTriggerTcpServer::~RecordTriggerTcpServer() { stop(); }

void RecordTriggerTcpServer::configure(bool enabled, quint16 port)
{
    const bool changed = enabled_ != enabled || port_ != port;
    enabled_ = enabled;
    port_ = port;

    if (!changed && (!enabled_ || server_.isListening()))
        return;

    stop();

    if (enabled_)
        start();
}

void RecordTriggerTcpServer::start()
{
    if (server_.isListening())
        return;

    if (!server_.listen(QHostAddress::LocalHost, port_))
    {
        LOG_WARN("MATLAB record trigger TCP server could not listen on 127.0.0.1:{}: {}",
                 port_,
                 server_.errorString().toStdString());
        return;
    }

    LOG_INFO("MATLAB record trigger TCP server listening on 127.0.0.1:{}", port_);
}

void RecordTriggerTcpServer::stop()
{
    const QVector<QTcpSocket*> clients = clients_;
    for (QTcpSocket* client : clients)
    {
        if (!client)
            continue;

        client->disconnectFromHost();
        client->deleteLater();
    }
    clients_.clear();

    if (server_.isListening())
        server_.close();
}

void RecordTriggerTcpServer::accept_pending_connections()
{
    while (QTcpSocket* client = server_.nextPendingConnection())
    {
        client->setParent(&server_);
        clients_.push_back(client);

        QObject::connect(client, &QTcpSocket::disconnected, &server_, [this, client]() { remove_client(client); });

        LOG_INFO("MATLAB record trigger client connected from {}:{}",
                 client->peerAddress().toString().toStdString(),
                 client->peerPort());
    }
}

void RecordTriggerTcpServer::remove_client(QTcpSocket* client)
{
    clients_.removeAll(client);

    if (client)
        client->deleteLater();
}

void RecordTriggerTcpServer::notify_record_started()
{
    if (!enabled_ || !server_.isListening())
        return;

    static constexpr char payload[] = "RECORD_START\n";

    for (QTcpSocket* client : clients_)
    {
        if (!client || client->state() != QAbstractSocket::ConnectedState)
            continue;

        client->write(payload, sizeof(payload) - 1);
        client->flush();
    }
}

} // namespace holovibes::gui
