#!/usr/bin/env ruby

require 'trollop'
require 'zipf'

def usage
  STDERR.write "Usage: "
  STDERR.write "ruby parallelize.rb -c <dtrain.ini> [-e <epochs=10>] [--randomize/-z] [--reshard/-y] -s <#shards|0> [-p <at once=9999>] -i <input> [--qsub/-q] [--dtrain_binary <path to dtrain binary>] [-l \"l2 select_k 100000\"] [--extra_qsub \"-l mem_free=24G\"]\n"
  exit 1
end

opts = Trollop::options do
  opt :config, "dtrain config file", :type => :string
  opt :epochs, "number of epochs", :type => :int, :default => 10
  opt :lplp_args, "arguments for lplp.rb", :type => :string, :default => "l2 select_k 100000"
  opt :randomize, "randomize shards before each epoch", :type => :bool, :short => '-z', :default => false
  opt :reshard, "reshard after each epoch", :type => :bool, :short => '-y', :default => false
  opt :shards, "number of shards", :type => :int
  opt :processes_at_once, "have this number (max) running at the same time", :type => :int, :default => 9999
  opt :input, "input (bitext f ||| e ||| ...)", :type => :string
  opt :dtrain_binary, "path to dtrain binary", :type => :string
  opt :qsub, "use qsub", :type => :bool, :default => false
  opt :qsub_args, "extra args for qsub", :type => :string, :default => "-l h_vmem=5G"
  opt :first_input_weights, "input weights for first iter", :type => :string, :default => '', :short => '-w'
  opt :per_shard_decoder_configs, "give special decoder config per shard", :type => :string, :short => '-o'
end
usage if not opts[:config]&&opts[:shards]&&opts[:input]

dtrain_dir = File.expand_path File.dirname(__FILE__)
if not opts[:dtrain_binary]
  dtrain_bin = "#{dtrain_dir}/dtrain"
else
  dtrain_bin = opts[:dtrain_binary]
end
ruby       = '/usr/bin/ruby'
lplp_rb    = "#{dtrain_dir}/lplp.rb"
lplp_args  = opts[:lplp_args]
cat        = '/bin/cat'

ini        = opts[:config]
epochs     = opts[:epochs]
rand       = opts[:randomize]
reshard    = opts[:reshard]
predefined_shards = false
per_shard_decoder_configs = false
if opts[:shards] == 0
  predefined_shards = true
  num_shards = 0
  per_shard_decoder_configs = true if opts[:per_shard_decoder_configs]
else
  num_shards = opts[:shards]
end
input = opts[:input]
use_qsub       = opts[:qsub]
shards_at_once = opts[:processes_at_once]
first_input_weights  = opts[:first_input_weights]

`mkdir work`

def make_shards(input, num_shards, epoch, rand)
  lc = `wc -l #{input}`.split.first.to_i
  index = (0..lc-1).to_a
  index.reverse!
  index.shuffle! if rand
  shard_sz = (lc / num_shards.to_f).round 0
  leftover = lc - (num_shards*shard_sz)
  leftover = 0 if leftover < 0
  in_f = File.new input, 'r'
  in_lines = in_f.readlines
  shard_in_files = []
  in_fns = []
  new_num_shards = 0
  0.upto(num_shards-1) { |shard|
    break if index.size==0
    new_num_shards += 1
    in_fn = "work/shard.#{shard}.#{epoch}.in"
    shard_in = File.new in_fn, 'w+'
    in_fns << in_fn
    0.upto(shard_sz-1) { |i|
      j = index.pop
      break if !j
      shard_in.write in_lines[j]
    }
    shard_in_files << shard_in
  }
  while leftover > 0
    j = index.pop
    shard_in_files[-1].write in_lines[j]
    leftover -= 1
  end
  shard_in_files.each do |f| f.close end
  in_f.close
  return in_fns, new_num_shards
end

input_files = []
if predefined_shards
  input_files = File.new(input).readlines.map {|i| i.strip }
  if per_shard_decoder_configs
    decoder_configs = File.new(opts[:per_shard_decoder_configs]).readlines.map {|i| i.strip}
  end
  num_shards = input_files.size
else
  input_files, num_shards = make_shards input, num_shards, 0, rand
end

0.upto(epochs-1) { |epoch|
  puts "epoch #{epoch+1}"
  pids = []
  input_weights = ''
  if epoch > 0 then input_weights = "--input_weights work/weights.#{epoch-1}" end
  weights_files = []
  shard = 0
  remaining_shards = num_shards
  while remaining_shards > 0
    shards_at_once.times {
      break if remaining_shards==0
      qsub_str_start = qsub_str_end = ''
      local_end = ''
      if use_qsub
        qsub_str_start = "qsub #{opts[:qsub_args]} -cwd -sync y -b y -j y -o work/out.#{shard}.#{epoch} -N dtrain.#{shard}.#{epoch} \""
        qsub_str_end = "\""
        local_end = ''
      else
        local_end = "2>work/out.#{shard}.#{epoch}"
      end
      if per_shard_decoder_configs
        cdec_cfg = "--decoder_config #{decoder_configs[shard]}"
      else
        cdec_cfg = ""
      end
      if first_input_weights!='' && epoch == 0
        input_weights = "--input_weights #{first_input_weights}"
      end
      pids << Kernel.fork {
        `#{qsub_str_start}#{dtrain_bin} -c #{ini} #{cdec_cfg} #{input_weights}\
          --bitext #{input_files[shard]}\
          --output work/weights.#{shard}.#{epoch}#{qsub_str_end} #{local_end}`
      }
      weights_files << "work/weights.#{shard}.#{epoch}"
      shard += 1
      remaining_shards -= 1
    }
    pids.each { |pid| Process.wait(pid) }
    pids.clear
  end
  `#{cat} work/weights.*.#{epoch} > work/weights_cat`
  `#{ruby} #{lplp_rb} #{lplp_args} #{num_shards} < work/weights_cat > work/weights.#{epoch}`
  if rand and reshard and epoch+1!=epochs
    input_files, num_shards = make_shards input, num_shards, epoch+1, rand
  end
}

`rm work/weights_cat`

