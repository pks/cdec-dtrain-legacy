#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <unistd.h>

#include <nanomsg/nn.h>
#include <nanomsg/pair.h>
#include "nn.hpp"

using namespace std;

void
recv(nn::socket& sock)
{
  char *buf = NULL;
  size_t sz = sock.recv(&buf, NN_MSG, 0);
  if (buf) {
    string translation(buf, buf+sz);
    cout << "received translation '" << translation << "'" << endl;
  }
}

void
send(nn::socket& sock, const string& msg)
{
  cout << "sending source '" << msg << "'" << endl;
  sock.send(msg.c_str(), msg.size()+1, 0);
}

void
loop(nn::socket& sock)
{
  int to = 100;
  sock.setsockopt(NN_SOL_SOCKET, NN_RCVTIMEO, &to, sizeof(to));
  for (string line; getline(cin, line);) {
    send(sock, line);
    sleep(1);
    recv(sock);
  }
}

int main(int argc, char const* argv[])
{
  nn::socket sock(AF_SP, NN_PAIR);
  //string url = "ipc:///tmp/network_decoder.ipc";
  string url = "tcp://127.0.0.1:60666";
  sock.connect(url.c_str());
  loop(sock);

  return 0;
}

