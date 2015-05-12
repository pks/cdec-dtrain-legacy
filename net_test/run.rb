#!/usr/bin/env ruby

require 'nanomsg'
require 'zipf'

work_dir = "work"
`mkdir -p #{work_dir}`

# start daemons
# extractor
extractor_url = 'tcp://127.0.0.1:60000'
extractor_sock = NanoMsg::PairSocket.new
puts "starting extractor .."
extractor = fork do
  exec "python -m cdec.sa.extract -c extract.ini --online -u -S '#{extractor_url}' &>#{work_dir}/extractor.out"
end
Process.detach extractor
extractor_sock.connect extractor_url
puts "> got #{extractor_sock.recv} from extractor"
`mkdir -p #{work_dir}/g`

# aligner
aligner_url = 'tcp://127.0.0.1:60001'
aligner_sock = NanoMsg::PairSocket.new
puts "starting aligner .."
aligner = fork do 
  exec "../word-aligner/net_fa --sock_url '#{aligner_url}' -f a/ef &>#{work_dir}/net_fa.out"
end
Process.detach aligner
aligner_sock.connect aligner_url
puts "> got #{aligner_sock.recv} from aligner"

# dtrain
dtrain_url = 'tcp://127.0.0.1:60002'
dtrain_sock = NanoMsg::PairSocket.new
puts "starting dtrain .."
dtrain = fork do
  exec "../training/dtrain/dtrain_net_interface -c dtrain.ini -o #{work_dir}/weights.final -a '#{dtrain_url}' &>#{work_dir}/dtrain.out"
end
Process.detach dtrain
dtrain_sock.connect dtrain_url
puts "> got #{dtrain_sock.recv} from dtrain"

puts ">>> all daemons ready\n\n"

i = 0
while line = STDIN.gets
  source, reference = splitpipe(line.strip)
  source.strip!; reference.strip!
    puts "source: '#{source}'"
    puts "reference: '#{reference}'"
  generate_grammar = "TEST ||| #{source} ||| #{work_dir}/g/#{i}.grammar"
    puts "[extractor] generate grammar: '#{generate_grammar}'"
  extractor_sock.send generate_grammar
  grammar = extractor_sock.recv.strip
  annotated_source = "<seg id=\"#{i}\" grammar=\"#{grammar}\"> #{source} </seg>"
    puts "[extractor] got grammar in '#{grammar}'"
  dtrain_translate = "act:translate ||| #{annotated_source}"
    puts "[dtrain] translate: '#{dtrain_translate}'"
  dtrain_sock.send dtrain_translate
  transl = dtrain_sock.recv
    puts "[dtrain] received translation: '#{transl}'"
  source_and_ref = "#{annotated_source} ||| #{reference}"
    puts "[dtrain] sending '#{source_and_ref}' for learning"
  dtrain_sock.send source_and_ref
  source_and_ref = "#{source} ||| #{reference}"
    puts "[aligner] sending '#{source_and_ref}' to force align"
  aligner_sock.send source_and_ref
  fa = aligner_sock.recv
    puts "[aligner] got alignment: #{fa}'"
  extractor_example = "TEST ||| #{source} ||| #{reference} ||| #{fa.lstrip.strip}"
    puts "[extractor] sending '#{extractor_example}' for learning"
  extractor_sock.send "TEST ||| #{source} ||| #{reference} ||| #{fa.lstrip.strip}"
    puts "[extractor] #{extractor_sock.recv}"
  i += 1
    puts "---"
end

# stopping daemons
puts "\nshutting down all daemons .."
aligner_sock.send("shutdown")
puts "> aligner is #{aligner_sock.recv}"
extractor_sock.send("shutdown")
puts "> extractor is #{extractor_sock.recv}"
dtrain_sock.send("shutdown")
puts "> dtrain is #{dtrain_sock.recv}"

