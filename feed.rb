#!/usr/bin/env ruby

require 'nanomsg'
require 'trollop'

conf = Trollop::options do
  opt :addr, "URL of socket", :type => :string, :short => '-a', :default => "tcp://127.0.0.1:31337"
end

sock = NanoMsg::PairSocket.new
addr = conf[:addr]
sock.connect addr
puts "< got #{sock.recv}"

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

