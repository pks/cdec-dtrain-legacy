#!/usr/bin/env ruby

require 'nanomsg'

port = 60666
sock = NanoMsg::PairSocket.new
addr = "tcp://127.0.0.1:#{port}"
#addr = "ipc:///tmp/network_decoder.ipc"
sock.connect addr

while true
  line = STDIN.gets
  if !line
    sock.send 'shutdown'
    break
  end
  sock.send line.strip
  sleep 1
  puts "got translation: #{sock.recv}\n\n"
end

