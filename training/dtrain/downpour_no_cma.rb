#!/usr/bin/env ruby

require 'trollop'
require 'zipf'
require 'socket'
require 'nanomsg'
require 'tempfile'

def l2_select v, k=10000
  return if v.size<=k
  min = v.values.map { |i| i.abs2 }.sort.reverse[k-1]
  v.delete_if { |k,v| v.abs2 < min }
end

conf = Trollop::options do
  opt :conf,          "dtrain configuration",           :type => :string, :required => true, :short => '-c'
  opt :input,         "input as bitext (f ||| e)",      :type => :string, :required => true, :short => '-i'
  opt :epochs,        "number of epochs",               :type => :int,    :default => 10,    :short => '-e'
  opt :learning_rate, "learning rate",                  :type => :float,  :default => 1.0,   :short => '-l'
  opt :slaves,        "number of parallel learners",    :type => :int,    :default => 1,     :short => '-p'
  opt :dtrain_binary, "path to dtrain_net binary",      :type => :string,                    :short => '-d'
  opt :shuffle,       "shuffle data before each epoch",                                      :short => '-z'
  opt :select_freq,   "l2 feature selection: frequency", :type => :int,    :default => 100,  :short => '-f'
  opt :select_k,      "l2 feature selection: k",         :type => :int,    :default => 0,    :short => '-k'
end

dtrain_conf   = conf[:conf]
input         = conf[:input]
epochs        = conf[:epochs]
learning_rate = conf[:learning_rate]
num_slaves    = conf[:slaves]
shuf          = conf[:shuffle]
freq          = conf[:select_freq]
k             = conf[:select_k]
select        = k>0
dtrain_dir    = File.expand_path File.dirname(__FILE__)

if not conf[:dtrain_binary]
  dtrain_bin = "#{dtrain_dir}/dtrain_net"
else
  dtrain_bin = conf[:dtrain_binary]
end

socks      = []
slave_pids = []
#port       = 60666 # last port = port+slaves
#master_ip  = Socket.ip_address_list[0].ip_address

`mkdir work`

socks_files = []
num_slaves.times { |i|
  socks << NanoMsg::PairSocket.new
  #addr = "tcp://#{master_ip}:#{port}"
  socks_files << Tempfile.new('downpour')
  url = "ipc://#{socks_files.last.path}"
  socks.last.bind url
  STDERR.write "listening on #{url}\n"
  slave_pids << Kernel.fork {
    `LD_LIBRARY_PATH=/scratch/simianer/downpour/nanomsg-0.5-beta/lib \
      #{dtrain_bin} -c #{dtrain_conf} -a #{url} 2>work/out.#{i}`
  }
  #port += 1
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

def shuffle_file fn_in, fn_out
  a = ReadFile.readlines_strip fn_in
  o = WriteFile.new fn_out
  a.shuffle!
  o.write a.join("\n")+"\n"
  o.close

  return fn_out
end

inf = nil
if shuf
  input = shuffle_file input, "work/input.0.gz"
  inf = ReadFile.new input
else
  inf = ReadFile.new input
end
buf = []
j = 0
m = Mutex.new
n = Mutex.new
w = SparseVector.new
ready = num_slaves.times.map { true }
cma = 1
epochs.times { |epoch|
STDERR.write "---\nepoch #{epoch}\n"
if shuf && epoch>0
  inf.close
  input = shuffle_file input, "work/input.#{epoch}.gz"
  inf = ReadFile.new input
else
  inf.rewind
end
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
  n.synchronize { ready[j] = false }
  threads << Thread.new {
    me = j
    moment = cma
    update = SparseVector::from_kv socks[me].recv
    STDERR.write "T update from slave ##{me}\n"
    update *= learning_rate
    #update -= w
    #update /= moment
    m.synchronize { w += update }
    STDERR.write "T sending new weights to slave ##{me}\n"
    socks[me].send w.to_kv
    STDERR.write "T sent new weights to slave ##{me}\n"
    n.synchronize { ready[me] = true }
  }
  sleep 1
  if select && i>0 && (i+1)%freq==0
    before = w.size
    m.synchronize { l2_select w, k }
    STDERR.write "l2 feature selection, before=#{before}, after=#{w.size}\n"
  end
  i += 1
  cma += 1
  j += 1
  j = 0 if j==num_slaves
  threads.delete_if { |thr| !thr.status }
end
wf = WriteFile.new "weights.#{epoch}.gz"
wf.write w.to_kv(" ", "\n")+"\n"
wf.close
}

threads.each { |thr| thr.join }

socks.each { |n|
  Thread.new {
    n.send "shutdown"
  }
}

slave_pids.each { |pid| Process.wait(pid) }

socks_files.each { |f| f.unlink }

