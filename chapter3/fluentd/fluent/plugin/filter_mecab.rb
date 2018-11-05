module Fluent
class MeCabFilter < Filter
  Plugin.register_filter('mecab', self)
  config_param :key, :string
  config_param :tag, :string, default: "mecab"

  def initialize
    super
    require 'natto'
  end

  def configure(config)
    super
    @mecab = Natto::MeCab.new
  end

  def start
    super
  end

  def shutdown
    super
  end

  def filter(tag, time, record)
  end

  def filter_stream(tag, es)
    result_es = MultiEventStream.new
    es.each do |time, record|
      begin
        position = 0
        @mecab.parse(pre_process(record[@key])) do |mecab|
          length = mecab.surface.length
          next if length == 0

          new_record = record.clone
          new_record["mecab"] = { "word" => mecab.surface,
                                  "length" => length,
                                  "pos"  => mecab.feature.split(/\,/),
                                  "position" => position}        
          result_es.add(time, new_record)

          position += length
        end
      rescue => e
        router.emit_error_event(tag, time, record, e)
      end
    end
    return result_es
  end

  def pre_process(text)
    # delete URL
    return text.gsub(/https?\:\/\/([\w\-]+\.)+[\w-]+(\/[\w-]+)*\/?/, '').gsub(/RT\s*:\*/, '').gsub(/@[\w]+\s*/, '')
  end
end
end
