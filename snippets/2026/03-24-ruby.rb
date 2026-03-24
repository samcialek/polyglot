# Singly linked list that mixes in Enumerable.

class LinkedList
  include Enumerable
  Node = Struct.new(:value, :next_node)

  def initialize
    @head = nil
    @size = 0
  end

  attr_reader :size

  def push(value)
    @head = Node.new(value, @head)
    @size += 1
    self
  end

  def pop
    return nil if @head.nil?
    val = @head.value
    @head = @head.next_node
    @size -= 1
    val
  end

  def each
    node = @head
    while node
      yield node.value
      node = node.next_node
    end
  end

  def to_s
    "[#{to_a.join(" -> ")}]"
  end
end

list = LinkedList.new
[5, 4, 3, 2, 1].each { |n| list.push(n) }
puts list
puts "Sum:  #{list.sum}"
puts "Max:  #{list.max}"
puts "Even: #{list.select(&:even?)}"
