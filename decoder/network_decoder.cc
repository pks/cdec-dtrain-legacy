#include <string>

#include "decoder.h"
#include "ff_register.h"
#include "filelib.h"
#include "verbose.h"
#include "viterbi.h"

#include <nanomsg/nn.h>
#include <nanomsg/pair.h>
#include "nn.hpp"

using namespace std;

struct TheObserver : public DecoderObserver
{
  string translation;

  virtual void
  NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg)
  {
    translation.clear();
    vector<WordID> trans;
    ViterbiESentence(*hg, &trans);
    translation = TD::GetString(trans);
  }
};

int send(nn::socket& sock, const string trans)
{
  cout << "sending translation '" << trans << "'" << endl;
  sock.send(trans.c_str(), trans.size()+1, 0);
}

bool
recv(nn::socket& sock, string& source)
{
  char *buf = NULL;
  sock.recv(&buf, NN_MSG, 0);
  if (buf) {
    string s(buf);
    source = s;

    return true;
  }

  return false;
}

void
loop(Decoder& decoder, nn::socket& sock)
{
  int to = 100;
  sock.setsockopt(NN_SOL_SOCKET, NN_RCVTIMEO, &to, sizeof (to));
  TheObserver o;

  while(true)
  {
    string source;
    bool r = recv(sock, source);
    if (r) {
      cout << "received source '" << source << "'" << endl;
      decoder.Decode(source, &o);
      send(sock, o.translation);
    }
  }
}
  
int
main(int argc, char** argv)
{
  register_feature_functions();
  ReadFile f(argv[1]);
  Decoder decoder(f.stream());
  SetSilent(true);

  nn::socket sock(AF_SP, NN_PAIR);
  string url = "ipc:///tmp/network_decoder.ipc";
  sock.bind(url.c_str());

  loop(decoder, sock);

  return 0;
}

