#!/usr/bin/env ruby

require 'trollop'
require 'zipf'
require 'socket'
require 'nanomsg'

conf = Trollop::options do
  opt :conf,          "dtrain configuration",        :type => :string, :required => true, :short => '-c'
  opt :input,         "input as bitext (f ||| e)",   :type => :string, :required => true, :short => '-i'
  opt :epochs,        "number of epochs",            :type => :int,    :default => 10,    :short => '-e'
  opt :learning_rate, "learning rate",               :type => :float,  :default => 1.0,   :short => '-l'
  opt :slaves,        "number of parallel learners", :type => :int,    :default => 1,     :short => '-p'
  opt :dtrain_binary, "path to dtrain_net binary",   :type => :string,                    :short => '-d'
end

dtrain_conf   = conf[:conf]
input         = conf[:input]
epochs        = conf[:epochs]
learning_rate = conf[:learning_rate]
num_slaves    = conf[:slaves]
dtrain_dir    = File.expand_path File.dirname(__FILE__)

if not conf[:dtrain_binary]
  dtrain_bin = "#{dtrain_dir}/dtrain_net"
else
  dtrain_bin = conf[:dtrain_binary]
end

socks      = []
port       = 60666 # last port = port+slaves
slave_pids = []
master_ip  = Socket.ip_address_list[0].ip_address

`mkdir work`

num_slaves.times { |i|
  socks << NanoMsg::PairSocket.new
  addr = "tcp://#{master_ip}:#{port}"
  socks.last.bind addr
  STDERR.write "listening on #{addr}\n"
  slave_pids << Kernel.fork {
    cmd = "#{dtrain_bin} -c #{dtrain_conf} -a #{addr} &>work/out.#{i}"
    `#{cmd}`
  }
  port += 1
}

threads = []
socks.each_with_index { |n,i|
  threads << Thread.new {
    n.recv
    STDERR.write "got hello from slave ##{i}\n"
  }
}
threads.each { |thr| thr.join } # timeout?
threads.clear

inf = ReadFile.new input
buf = []
j = 0
m = Mutex.new
n = Mutex.new
w = SparseVector.new
ready = num_slaves.times.map { true }
cma = 1
epochs.times { |epoch|
STDERR.write "---\nepoch #{epoch}\n"
inf.rewind
i = 0
while true # round-robin
  d = inf.gets
  break if !d
  d.strip!
  while !ready[j]
    j += 1
    j = 0 if j==num_slaves
  end
  STDERR.write "sending source ##{i} to slave ##{j}\n"
  socks[j].send d
  n.synchronize {
    ready[j] = false
  }
  threads << Thread.new {
    me = j
    moment = cma
    update = SparseVector::from_kv socks[me].recv
    STDERR.write "T update from slave ##{me}\n"
    update *= learning_rate
    update -= w
    update /= moment
    m.synchronize { w += update }
    STDERR.write "T sending new weights to slave ##{me}\n"
    socks[me].send w.to_kv
    STDERR.write "T sent new weights to slave ##{me}\n"
    n.synchronize {
      ready[me] = true
    }
  }
  sleep 1
  i += 1
  cma += 1
  j += 1
  j = 0 if j==num_slaves
  threads.delete_if { |thr| !thr.status }
end
}

threads.each { |thr| thr.join }

socks.each { |n|
  Thread.new {
    n.send "shutdown"
  }
}

slave_pids.each { |pid| Process.wait(pid) }

puts w.to_kv " ", "\n"

