#!/usr/bin/env ruby

require 'nanomsg'

port = ARGV[0]
sock = NanoMsg::PairSocket.new
addr = "tcp://127.0.0.1:#{port}"
sock.bind addr

puts sock.recv
while true
  line = STDIN.gets
  if !line
    sock.send 'shutdown'
    break
  end
  sock.send line.strip
  sleep 1
  sock.recv
  sock.send "a=1 b=2"
end

