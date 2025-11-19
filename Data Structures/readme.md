
***

# Data Structures: Overview & Cheat Sheet

Data structures are fundamental tools in computer science for organizing, storing, and managing data efficiently. Mastering these structures is essential for developing high-performance software and algorithms.

## Comparison Table

| Data Structure      | Description                                                               | Common Use Cases                  | Main Operations (Big O)              | Advantages                            | Disadvantages                         | Top Resources   |
|-------------------- |:--------------------------------------------------------------------------|:----------------------------------|:--------------------------------------|:--------------------------------------|:--------------------------------------|:---------------|
| **Array**           | Fixed-size, contiguous items, index based                                 | Static datasets, fast access      | Access: O(1)<br>Insert: O(n)<br>Delete: O(n)   | Fast lookup, simple structure          | Size fixed, slow insert/delete         | [GFG Array](https://www.geeksforgeeks.org/array-data-structure/) |
| **Dynamic Array**   | Auto-resizing array (Python list, C++ vector)                            | Flexible, growing data sets       | Access: O(1)<br>Append: O(1)*<br>Insert: O(n) | Flexible size, O(1) access             | Occasional O(n) resize on append       | [Adrian Mejia](https://adrianmejia.com/data-structures-time-complexity-for-beginners-arrays-hashmaps-linked-lists-stacks-queues-tutorial/) |
| **Linked List**     | Nodes with value & pointer (singly, doubly, circular)                    | Playlists, buffer, undo, queues   | Access: O(n)<br>Insert: O(1)<br>Delete: O(1)  | Easy insert/delete, dynamic sizing      | Slow search, more memory than array    | [AfterAcademy](https://afteracademy.com/blog/introduction-to-data-structure) |
| **Stack**           | LIFO (last-in, first-out) structure                                      | Undo, parsing, recursion          | Push: O(1)<br>Pop: O(1)<br>Peek: O(1)         | Simple, efficient                      | Only top access                        | [GFG Stack](https://www.geeksforgeeks.org/stack-data-structure/)  |
| **Queue**           | FIFO (first-in, first-out) structure                                     | Buffers, scheduling, tasks        | Enqueue: O(1)<br>Dequeue: O(1)                | Simple, fair order                     | Only front/rear access                 | [GFG Queue](https://www.geeksforgeeks.org/queue-data-structure/)  |
| **Hash Table / Dict**| Key-value pair lookup via hashing                                       | Fast dictionary/map, caching      | Insert: O(1)<br>Search: O(1)<br>Delete: O(1)  | Fast, flexible keys                     | Poor in collision-heavy cases          | [Adrian Mejia](https://adrianmejia.com/data-structures-time-complexity-for-beginners-arrays-hashmaps-linked-lists-stacks-queues-tutorial/) |
| **Binary Tree**     | Tree, ≤2 children per node                                               | Hierarchical data, parse trees    | Search/Insert: O(n)\*                         | Recursive, structured                   | Can become unbalanced                  | [VisualGo BST](https://visualgo.net/en/bst) |
| **BST**             | Ordered binary tree (left < node < right)                                | Fast search, sets, maps           | Search: O(log n)\*<br>Insert: O(log n)\*      | Fast ops if balanced                    | Unbalanced trees slow (O(n))           | [VisualGo BST](https://visualgo.net/en/bst) |
| **AVL Tree**        | Self-balancing BST                                                       | Sorted collections, DB indexes    | Search/Insert: O(log n)                        | Always balanced, reliable speed         | Extra rotations on insert/delete        | [GFG AVL](https://www.geeksforgeeks.org/dsa/introduction-to-avl-tree/) |
| **B-Tree**          | Multiway balanced search tree                                            | Databases, file systems           | All ops: O(log n)                                 | Efficient with disks, broad nodes       | Complex logic                          | [GFG B-Tree](https://www.geeksforgeeks.org/b-tree-set-1-introduction-2/) |
| **Heap**            | Complete binary tree, parent > (max) or < (min) children                 | Priority queue, scheduling       | Insert: O(log n)<br>Extract: O(log n)           | Easy max/min extraction, fast           | No fast arbitrary search               | [GFG Heap](https://www.geeksforgeeks.org/heap-data-structure/) |
| **Priority Queue**  | Each item assigned a priority                                            | Scheduling, Dijkstra’s algorithm | Insert: O(log n)<br>Extract: O(log n)           | Always get highest priority             | Can’t quickly find arbitrary values     | [GFG Priority Queue](https://www.geeksforgeeks.org/dsa/how-to-implement-priority-queue-using-heap-or-array/) |
| **Graph**           | Nodes & edges, may be directed or weighted                               | Web, map navigation, social nets  | Depends: List/Matrix, traversal varies           | Abstract connections, model networks    | Complex logic, memory                   | [Study.com Graphs](https://study.com/academy/lesson/weighted-graphs-implementation-dijkstra-algorithm.html)  |
| **Big O Notation**  | Describes efficiency as data grows                                       | Algorithm analysis                | -                                               | Universal for comparing performance     | Only worst-case, not always precise     | [GFG Big O](https://www.geeksforgeeks.org/dsa/analysis-algorithms-big-o-analysis/) |

> \* O(1) = constant time, O(log n) = logarithmic, O(n) = linear. Amortized O(1) applies to dynamic array append; log n is typical if BST/heap/tree is balanced.

***

## Data Structures Explained

### 1. Arrays & Dynamic Arrays
- Array: Fixed size, quick random access. Great for static data.
- Dynamic Array: Grows/shrinks as needed. Used for flexible lists, e.g., Python's `list`.

### 2. Linked Lists
- Nodes connect by pointers.
- **Single**: One way; **Double**: Both ways; **Circular**: Loops back.
- Use when frequent insertion/deletion required.

### 3. Stack & Queue
- Stack: LIFO, supports quick push/pop.
- Queue: FIFO, elements exit in the order entered.
- Key for algorithm parsing, task scheduling.

### 4. Hash Table / Dictionary
- Maps keys to values with hashing.
- Lightning-fast lookups, used in caches and object stores.

### 5. Trees
- **Binary Tree**: Each node max 2 children.
- **BST**: Sorted property, for efficient search.
- **AVL**: BST that auto-balances for steady speed.
- **B-Tree**: Multiway tree, ideal for storage and databases.

### 6. Graphs
- Vertices (nodes) with edges.
- Can be directional or weighted.
- Model relationships and complex systems, like maps or social networks.

### 7. Heaps & Priority Queues
- Heap: Tree where root always higher (or lower) than children.
- Priority Queue: Top-priority element always accessible.
- Used in job scheduling, pathfinding (like Dijkstra’s algorithm).

### 8. Big O Notation
- Standard for analyzing scaling/performance.
- E.g.: O(1) = fast, O(n) = slow as data grows.
- Crucial for picking the right structure for your needs.

## Best Platforms to Learn & Practice

- [GeeksforGeeks Data Structures Tutorials](https://www.geeksforgeeks.org/data-structures/)  
- [NeetCode - DSA Patterns for Interviews](https://neetcode.io/)  
- [VisualGo - Visualize DS & Alg](https://visualgo.net/en)  
- [LeetCode - Real Practice](https://leetcode.com/)  
- [Adrian Mejia - Beginner DS Guide](https://adrianmejia.com/data-structures-time-complexity-for-beginners-arrays-hashmaps-linked-lists-stacks-queues-tutorial/)  

***

**Tips:**  
- Don’t just read—code each structure yourself!  
- Visualize (see [VisualGo](https://visualgo.net/en)).  
- Watch out for trade-offs in speed, flexibility, and memory.  
- Use coding platforms ([LeetCode](https://leetcode.com/), [HackerRank](https://www.hackerrank.com/domains/tutorials/10-days-of-javascript)) for challenge-based practice.

***
