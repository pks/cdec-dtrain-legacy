#!/usr/bin/env ruby

require 'nanomsg'

port = 8888
sock = NanoMsg::PairSocket.new
addr = "tcp://127.0.0.1:#{port}"
sock.connect addr
puts sock.recv

while true
  line = STDIN.gets
  if !line || line.strip=='shutdown'
    puts "shutting down"
    sock.send 'shutdown'
    break
  end
  puts "sending '#{line.strip}'"
  sock.send line.strip
  sleep 1
  puts "got response '#{sock.recv}'"
end

