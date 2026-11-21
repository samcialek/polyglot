import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class GenericStack<T> implements Iterable<T> {
    @SuppressWarnings("unchecked")
    private T[] data = (T[]) new Object[4];
    private int size = 0;

    public void push(T item) {
        if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
        data[size++] = item;
    }

    public T pop() {
        if (size == 0) throw new NoSuchElementException();
        T item = data[--size]; data[size] = null; return item;
    }

    public T peek() {
        if (size == 0) throw new NoSuchElementException();
        return data[size - 1];
    }

    public int size() { return size; }

    @Override
    public Iterator<T> iterator() {
        return new Iterator<>() {
            private int i = size - 1;
            public boolean hasNext() { return i >= 0; }
            public T next() { return data[i--]; }
        };
    }

    public static void main(String[] args) {
        GenericStack<Integer> stack = new GenericStack<>();
        for (int i = 1; i <= 5; i++) stack.push(i);
        System.out.println("Peek: " + stack.peek());
        for (int val : stack) System.out.print(val + " ");
        System.out.println();
    }
}
