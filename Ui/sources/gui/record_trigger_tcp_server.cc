#include "record_trigger_tcp_server.hh"

#include <utility>

#include <QHostAddress>

#include "logger.hh"

namespace holovibes::gui
{
namespace
{
constexpr char RECORD_START_COMMAND[] = "RECORD_START";
constexpr char RECORD_START_PAYLOAD[] = "RECORD_START\n";
constexpr qsizetype MAX_PENDING_COMMAND_SIZE = 256;
} // namespace

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
        LOG_WARN("Record trigger TCP server could not listen on 127.0.0.1:{}: {}",
                 port_,
                 server_.errorString().toStdString());
        return;
    }

    LOG_INFO("Record trigger TCP server listening on 127.0.0.1:{}", port_);
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
    receive_buffers_.clear();

    if (server_.isListening())
        server_.close();
}

void RecordTriggerTcpServer::accept_pending_connections()
{
    while (QTcpSocket* client = server_.nextPendingConnection())
    {
        client->setParent(&server_);
        clients_.push_back(client);
        receive_buffers_.insert(client, QByteArray{});

        QObject::connect(client, &QTcpSocket::readyRead, &server_, [this, client]() { process_client_data(client); });
        QObject::connect(client, &QTcpSocket::disconnected, &server_, [this, client]() { remove_client(client); });

        LOG_INFO("Record trigger TCP client connected from {}:{}",
                 client->peerAddress().toString().toStdString(),
                 client->peerPort());

        if (client->bytesAvailable() > 0)
            process_client_data(client);
    }
}

void RecordTriggerTcpServer::process_client_data(QTcpSocket* client)
{
    if (!client || !receive_buffers_.contains(client))
        return;

    QByteArray& buffer = receive_buffers_[client];
    buffer.append(client->readAll());

    qsizetype newline_index = 0;
    while ((newline_index = buffer.indexOf('\n')) >= 0)
    {
        const QByteArray command = buffer.left(newline_index).trimmed();
        buffer.remove(0, newline_index + 1);

        if (command.isEmpty())
            continue;

        if (command.size() > MAX_PENDING_COMMAND_SIZE)
        {
            LOG_WARN("Record trigger TCP server discarded a command longer than {} bytes", MAX_PENDING_COMMAND_SIZE);
            continue;
        }

        if (command != RECORD_START_COMMAND)
        {
            LOG_WARN("Record trigger TCP server received unknown command: {}", command.toStdString());
            continue;
        }

        LOG_INFO("Record trigger TCP server received a recording start request");
        if (record_start_callback_)
            record_start_callback_();
        else
            LOG_WARN("Record trigger TCP server cannot start a recording: no callback is registered");
    }

    if (buffer.size() > MAX_PENDING_COMMAND_SIZE)
    {
        LOG_WARN("Record trigger TCP server discarded an unterminated command longer than {} bytes",
                 MAX_PENDING_COMMAND_SIZE);
        buffer.clear();
    }
}

void RecordTriggerTcpServer::remove_client(QTcpSocket* client)
{
    clients_.removeAll(client);
    receive_buffers_.remove(client);

    if (client)
        client->deleteLater();
}

void RecordTriggerTcpServer::set_record_start_callback(std::function<void()> callback)
{
    record_start_callback_ = std::move(callback);
}

void RecordTriggerTcpServer::notify_record_started()
{
    if (!enabled_ || !server_.isListening())
        return;

    for (QTcpSocket* client : clients_)
    {
        if (!client || client->state() != QAbstractSocket::ConnectedState)
            continue;

        client->write(RECORD_START_PAYLOAD, sizeof(RECORD_START_PAYLOAD) - 1);
        client->flush();
    }
}

} // namespace holovibes::gui
