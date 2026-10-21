public class BinaryTree {
    private record Node(int value, Node left, Node right) {
        Node withLeft(Node l) { return new Node(value, l, right); }
        Node withRight(Node r) { return new Node(value, left, r); }
    }

    private Node root;

    public void insert(int value) { root = insert(root, value); }

    private Node insert(Node node, int value) {
        if (node == null) return new Node(value, null, null);
        if (value < node.value) return node.withLeft(insert(node.left, value));
        if (value > node.value) return node.withRight(insert(node.right, value));
        return node;
    }

    public boolean contains(int value) { return contains(root, value); }

    private boolean contains(Node node, int value) {
        if (node == null) return false;
        if (value == node.value) return true;
        return value < node.value ? contains(node.left, value) : contains(node.right, value);
    }

    public void inOrder() { inOrder(root); System.out.println(); }

    private void inOrder(Node node) {
        if (node == null) return;
        inOrder(node.left);
        System.out.print(node.value + " ");
        inOrder(node.right);
    }

    public static void main(String[] args) {
        BinaryTree tree = new BinaryTree();
        for (int v : new int[]{5, 3, 7, 1, 4, 6, 8}) tree.insert(v);
        tree.inOrder();
        System.out.println("Contains 4: " + tree.contains(4));
    }
}
