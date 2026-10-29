# Ruby Enumerable methods with practical examples.

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

evens, odds = data.partition(&:even?)
puts "Evens: #{evens}"
puts "Odds:  #{odds}"

grouped = data.group_by { |n| n < 5 ? :small : :large }
puts "Grouped: #{grouped}"

averages = data.each_with_object([]) do |n, acc|
  prev_sum = acc.empty? ? 0 : acc.last[:sum]
  prev_cnt = acc.empty? ? 0 : acc.last[:count]
  acc << { sum: prev_sum + n, count: prev_cnt + 1,
           avg: (prev_sum + n).to_f / (prev_cnt + 1) }
end
puts "Running averages: #{averages.map { |a| a[:avg].round(2) }}"

words = ["hello world", "hello ruby", "world class"]
freq = words.flat_map { |s| s.split }.tally
puts "Word freq: #{freq}"
