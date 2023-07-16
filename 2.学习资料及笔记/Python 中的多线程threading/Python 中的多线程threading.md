# Python 中的多线程threading

在Python中，线程是一种可以在单个进程中进行并发执行的技术。使用线程，你可以同时运行多个任务，这样可以有效地利用多核处理器并提高程序的效率。

Python的**`threading`**模块是内置的，提供了用于创建和管理线程的功能。

以下是一个例子：

```python
import threading

def print_numbers():
    for i in range(10):
        print(i)

def print_letters():
    for letter in 'abcdefghij':
        print(letter)

# 创建两个线程
thread1 = threading.Thread(target=print_numbers)
#如果是有传参的函数，可以写成元组形式
#threading.Thread(target=print_numbers, args=('your_args',))
thread2 = threading.Thread(target=print_letters)

# 启动线程
thread1.start()
thread2.start()

# 等待这两个线程执行完成
thread1.join()
thread2.join()
```

输出为以下：

```python
0
1
2
3
4
5
6
7
8
9
a
b
c
d
e
f
g
h
i
j
```

如果不希望两个线程按顺序执行，可以将 `.join()` 注释掉

```python
0
1a

b2

c3

4d

5e
6

f7
g

h8

i
9
j
```

然而，需要注意的是，由于Python的全局解释器锁（GIL）的存在，Python的多线程并不能实现真正意义上的并行处理（在同一时间点，多个任务同时进行）。在任何时候，只有一个线程在执行，而其他线程在等待GIL的释放。因此，在涉及到CPU密集型任务（例如大量计算）时，Python的多线程可能并不会提高程序的效率。在这种情况下，可以考虑使用Python的**`multiprocessing`**模块，该模块使用子进程而不是线程，每个子进程有自己的独立的GIL，因此可以真正地并行执行。