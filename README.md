# 作业 3：简易 CUDA 渲染器 (A Simple CUDA Renderer)

[English Version / 英文原版](README_en.md)

**截止时间：10 月 30 日（周四）晚上 11:59 PST**  
**总分：100 分**

![My Image](handout/teaser.jpg?raw=true)

## 概述 (Overview)

在本次作业中，你将使用 CUDA 编写一个绘制彩色圆形的并行渲染器。虽然这个渲染器非常简朴，但为了将其并行化，你需要设计并实现能够在并行环境下高效构建和操作的数据结构。这是一项富有挑战性的作业，因此强烈建议你**尽早开始**。**认真地说，请尽早动手！** 祝你好运！

## 环境设置 (Environment Setup)

1. 你将在亚马逊云服务（AWS）配备 GPU 的虚拟机实例上收集本次作业的结果（即运行性能测试）。请按照 [cloud_readme.md](cloud_readme.md) 中的说明设置运行机器。
2. 从课程 Github 下载作业初始代码：
   ```bash
   git clone https://github.com/stanford-cs149/asst3
   ```

CUDA C 编程指南的 [PDF 版本](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) 或 [网页版](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 是学习 CUDA 编程的优秀参考资料。网络上（直接 Google！）以及 [NVIDIA 开发者网站](http://docs.nvidia.com/cuda/) 上有大量的 CUDA 教程和 SDK 示例。特别是，你可能会喜欢 Udacity 上的免费课程 [CUDA 并行编程导论 (Introduction to Parallel Programming in CUDA)](https://www.udacity.com/blog/2014/01/update-on-udacity-cs344-intro-to.html)。

[CUDA C 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) 中的表 21 是关于本次作业所使用的 NVIDIA T4 GPU 的每个线程块最大 CUDA 线程数、线程块大小、共享内存大小等硬件参数的便捷参考表。NVIDIA T4 GPU 支持 CUDA 计算能力（Compute Capability）7.5。

关于 C++ 疑问（例如 *virtual* 关键字的含义），[C++ Super-FAQ](https://isocpp.org/faq) 是一个非常优秀的资源，它以详细且通俗易懂的方式进行解释，并由 C++ 之父 Bjarne Stroustrup 共同撰写！

---

## 第一部分：CUDA 热身 1：SAXPY (Part 1: CUDA Warm-Up 1: SAXPY - 5分)

为了熟悉 CUDA 程序的编写，你的热身任务是用 CUDA 重新实现作业 1 中的 SAXPY 函数。此部分的初始代码位于作业仓库的 `/saxpy` 目录下。你可以通过在 `/saxpy` 目录下执行 `make` 和 `./cudaSaxpy` 来构建并运行 saxpy CUDA 程序。

请在 `saxpy.cu` 的 `saxpyCuda` 函数中完成 SAXPY 的实现。在执行计算之前，你需要分配设备全局内存数组，并将主机输入数组 `X`、`Y` 和 `result` 的内容复制到 CUDA 设备内存中。CUDA 计算完成后，必须将结果复制回主机内存。请参阅编程指南（网页版）3.2.2 节中 `cudaMemcpy` 函数的定义，或查看作业初始代码中指向的有用教程。

作为实现的一部分，请在 `saxpyCuda` 中的 CUDA 核函数调用前后添加计时器。添加完成后，你的程序应该测量两次执行时间：

- 提供的初始代码已包含计时器，用于测量**整个过程**的时间：将数据复制到 GPU、运行核函数、以及将数据复制回 CPU。
- 你还应插入计时器来**仅测量运行核函数的时间**（不应包含 CPU 到 GPU 的数据传输时间，也不应包含结果从 GPU 传输回 CPU 的时间）。

**在后一种情况下添加计时代码时，需要格外小心：** 默认情况下，GPU 上的 CUDA 核函数执行与 CPU 上运行的主应用程序线程是**异步**的。例如，如果你编写的代码如下：

```cpp
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
double endTime = CycleTimer::currentSeconds();
```

你测得的核函数执行时间会看起来惊人地快！因为你只测量了 API 调用本身的开销，而没有测量在 GPU 上实际执行计算的开销。

因此，你需要在核函数调用之后加上 `cudaDeviceSynchronize()` 调用，以等待 GPU 上的所有 CUDA 工作完成。当 GPU 上之前的所有 CUDA 工作完成时，`cudaDeviceSynchronize()` 才会返回。请注意，在 `cudaMemcpy()` 之后不需要调用 `cudaDeviceSynchronize()` 来确保传输到 GPU 完成，因为在我们当前使用的条件下 `cudaMemcpy()` 本身就是同步的（想了解更多细节，请参阅[此文档](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync)）。

```cpp
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
cudaDeviceSynchronize();
double endTime = CycleTimer::currentSeconds();
```

注意：在包含往返 CPU 传输时间的完整计时中，最终计时器之前（在将数据返回 CPU 的 `cudaMemcpy()` 调用之后）**不需要**调用 `cudaDeviceSynchronize()`，因为 `cudaMemcpy()` 在复制完成之前不会返回给调用线程。

**问题 1 (Question 1):** 与基于 CPU 的顺序执行 SAXPY 实现相比（回顾作业 1 中程序 5 的 saxpy 结果），你观察到了什么样的性能表现？

**问题 2 (Question 2):** 比较并解释两组计时器提供的结果差异（仅测量核函数执行时间 vs 测量包含数据在 CPU/GPU 间传输及核函数执行的整个过程）。观察到的带宽数值与机器各组件宣称的可用带宽是否**大致**一致？（你应该利用网络查找 NVIDIA T4 GPU 的内存带宽。提示：<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf>。AWS 上预期的内存总线带宽约为 5.3 GB/s，这与 16 通道 [PCIe 3.0](https://en.wikipedia.org/wiki/PCI_Express) 的理论值不匹配。造成无法达到峰值带宽的因素有多个，包括 CPU 主板芯片组性能，以及作为传输源的主机 CPU 内存是否为“固定页内存 (pinned memory)”——后者允许 GPU 直接访问内存而无需经过虚拟内存地址转换。如果感兴趣，可以在这里找到更多信息：<https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory>）。

---

## 第二部分：CUDA 热身 2：并行前缀和 (Part 2: CUDA Warm-Up 2: Parallel Prefix-Sum - 10分)

现在你已经熟悉了 CUDA 程序的基本结构与布局。作为第二个练习，你需要实现 `find_repeats` 函数的并行版本：给定一个整数列表 `A`，返回满足 `A[i] == A[i+1]` 的所有下标 `i` 的列表。

例如，给定数组 `{1,2,2,1,1,1,3,5,3,3}`，你的程序应输出数组 `{1,3,4,8}`。

#### 排他性前缀和 (Exclusive Prefix Sum)

我们希望你通过先实现并行排他性前缀和（Exclusive Prefix-Sum，也称为 Exclusive Scan）操作来完成 `find_repeats`。

排他性前缀和接收一个数组 `A`，并生成一个新数组 `output`，使得在每个下标 `i` 处的元素是截止到但**不包含** `A[i]` 的所有先前元素之和。例如，给定数组 `A = {1,4,6,8,2}`，排他性前缀和的输出为 `output = {0,1,5,11,19}`。

以下“类 C 代码”是 scan 的迭代版本。在下面的伪代码中，我们使用 `parallel_for` 表示潜在的并行循环。这与我们在课堂上讨论的算法相同：<https://gfxcourses.stanford.edu/cs149/fall25/lecture/dataparallel/slide_17>

```c
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // 上推阶段 (upsweep / reduce phase)
    for (int two_d = 1; two_d <= N/2; two_d *= 2) {
        int two_dplus1 = 2 * two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i + two_dplus1 - 1] += output[i + two_d - 1];
        }
    }

    output[N-1] = 0;

    // 下推阶段 (downsweep phase)
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2 * two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i + two_d - 1];
            output[i + two_d - 1] = output[i + two_dplus1 - 1];
            output[i + two_dplus1 - 1] += t;
        }
    }
}
```

我们希望你使用该算法在 CUDA 中实现并行前缀和。你必须在 `scan/scan.cu` 中实现 `exclusive_scan` 函数。你的实现将包含主机端和设备端代码，并需要多次启动 CUDA 核函数（对应上面伪代码中的每个 parallel_for 循环各启动一次）。

**注意：** 在初始代码中，上面的参考实现假设输入数组的长度 `N` 是 2 的幂次方。在 `cudaScan` 函数中，我们在 GPU 上分配对应缓冲区时将输入数组长度向上舍入为下一个 2 的幂次方以解决此问题。但是，代码只将 `N` 个元素从 GPU 缓冲区复制回 CPU 缓冲区。这个细节应该能简化你的 CUDA 实现。

编译后将生成可执行程序 `cudaScan`。命令行用法如下：

```text
Usage: ./cudaScan [options]

Program Options:
  -m  --test <TYPE>      Run specified function on input.  Valid tests are: scan, find_repeats (default: scan)
  -i  --input <NAME>     Run test on given input type. Valid inputs are: ones, random (default: random)
  -n  --arraysize <INT>  Number of elements in arrays
  -t  --thrust           Use Thrust library implementation
  -?  --help             This message
```

#### 使用前缀和实现“查找重复项” (Implementing "Find Repeats" Using Prefix Sum)

完成 `exclusive_scan` 之后，在 `scan/scan.cu` 中实现函数 `find_repeats`。这将涉及编写额外的设备代码，以及一次或多次调用 `exclusive_scan()`。你的代码应该将重复元素的下标列表写入提供的输出指针（在设备内存中），然后返回输出列表的大小。

在调用你的 `exclusive_scan` 实现时，请记住：`start` 数组的内容会被复制到 `output` 数组中。此外，传递给 `exclusive_scan` 的数组均假定位于**设备内存**中。

**评分机制：** 我们将在随机输入数组上测试你代码的正确性和性能。

作为参考，下面提供了一个 Scan 得分表示例，显示了在 K80 GPU 上简单 CUDA 实现的性能。要检查你的 `scan` 和 `find_repeats` 实现的正确性和性能得分，请分别运行 **`./checker.py scan`** 和 **`./checker.py find_repeats`**。运行后将输出如下所示的对照表；你的得分完全取决于你代码的性能。为了获得满分，你的代码性能必须在所提供参考解法的 20% 以内。

```text
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
```

这部分作业主要是为了让你多加练习编写 CUDA 代码并建立数据并行思维，而不是进行极致的性能微调。在这部分获得全部性能分不需要太多（或者说根本不需要）性能调优，只需将算法伪代码直接移植到 CUDA 即可。但是，有一个**关键技巧**：幼稚的 scan 实现可能会为伪代码中并行循环的每次迭代都启动 N 个 CUDA 线程，并在核函数内使用条件分支判断哪些线程实际需要工作。这种解法性能会很差！（想想上推阶段的最后一次最外层循环迭代，只有 2 个线程在干活！）。满分解法应只为最内层并行循环的实际迭代次数启动相应数量的 CUDA 线程。

**测试工具 (Test Harness)：** 默认情况下，测试工具在伪随机生成的数组上运行，该数组在每次运行程序时都相同，以便于调试。你可以传递 `-i random` 参数在随机数组上运行——我们在评分时会这么做。我们鼓励你构造其他输入来帮助评估程序。你也可以使用 `-n <size>` 选项更改输入数组的长度。

参数 `--thrust` 将使用 [Thrust 库](http://thrust.github.io/) 的 [exclusive scan 实现](https://docs.nvidia.com/cuda/archive/12.2.2/thrust/index.html?highlight=group%20prefix%20sums#prefix-sums)。**如果有人能做出可与 Thrust 媲美的实现，最多可获 2 分加分（Extra Credit）。**

---

## 第三部分：简易圆形渲染器 (Part 3: A Simple Circle Renderer - 85分)

重头戏来了！

作业初始代码的 `/render` 目录包含了一个绘制彩色圆形的渲染器实现。编译代码，并使用以下命令行运行渲染器：`./render -r cpuref rgb`。该程序将输出一个包含三个圆形的图像 `output_0000.ppm`。现在使用命令行 `./render -r cpuref snow` 运行渲染器，输出的图像将是飘落的雪花。在 macOS 上可以通过 Preview 直接查看 PPM 图像，在 Windows 上可能需要下载查看器。

注意：你也可以使用 `-i` 选项将渲染器输出发送到屏幕显示而不是文件（在 snow 场景下，你将看到雪花飘落的动画）。但是要使用交互模式，你需要在本地机器配置 X-windows 转发（[参考资料 1](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html) 或 [参考资料 2](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine) 可能会有帮助）。

作业初始代码包含两个版本的渲染器：在 `refRenderer.cpp` 中实现的串行、单线程 C++ 参考实现，以及在 `cudaRenderer.cu` 中实现的**有错误**的并行 CUDA 实现。

### 渲染器总览 (Renderer Overview)

我们鼓励你通过检查 `refRenderer.cpp` 中的参考实现来熟悉渲染器代码库的结构。方法 `setup` 在渲染第一帧之前被调用。在你的 CUDA 加速渲染器中，此方法可能包含所有渲染器初始化代码（分配缓冲区等）。每一帧都会调用 `render`，负责将所有圆形绘制到输出图像中。渲染器的另一个主函数 `advanceAnimation` 也是每帧调用一次，它负责更新圆形的位置和速度。在本次作业中你**不需要**修改 `advanceAnimation`。

渲染器接受一个圆形数组（3D 位置、速度、半径、颜色）作为输入。渲染每一帧的基本串行算法如下：

    清空图像 (Clear image)
    对于每个圆形:
        更新位置和速度
    对于每个圆形:
        计算屏幕外接矩形 (screen bounding box)
        对于外接矩形内的所有像素:
            计算像素中心点
            如果中心点在圆形内:
                计算该点处圆形的颜色
                将圆形的颜色贡献混合 (blend) 到该像素的图像中

下图说明了使用“点在圆内测试 (point-in-circle tests)”计算圆与像素覆盖率的基本算法。请注意：只有当像素的中心位于圆形内部时，圆形才会向输出像素贡献颜色。

![Point in circle test](handout/point_in_circle.jpg?raw=true "计算圆形对输出图像贡献的基本算法：对圆形包围盒内的所有像素进行覆盖测试。对于包围盒内的每个像素，如果其中心点（黑点）包含在圆形内，则认为该像素被圆形覆盖。圆形内部的像素中心标记为红色。仅对被覆盖的像素计算圆形对图像的贡献。")

渲染器的一个重要细节是它渲染的是**半透明**的圆。因此，任意一个像素的颜色并不是单个圆形的颜色，而是重叠在该像素上的所有半透明圆形贡献的混合结果（注意上面伪代码中的“混合贡献 (blend contribution)”部分）。渲染器使用包含红（R）、绿（G）、蓝（B）和不透明度（alpha）值的四元组（RGBA）表示圆形的颜色。Alpha = 1 对应完全不透明的圆，Alpha = 0 对应完全透明的圆。要在颜色为 `(P_r, P_g, P_b)` 的像素上方绘制颜色为 `(C_r, C_g, C_b, C_alpha)` 的半透明圆形，渲染器使用以下数学公式：

```text
result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
```

请注意，图像合成**不满足交换律**（物体 X 在 Y 之上与物体 Y 在 X 之上看起来并不相同），因此渲染器必须按照应用程序提供的顺序来绘制圆形（你可以假设应用程序按深度顺序提供圆形）。例如，考虑下面的两幅图像：一个蓝色圆画在一个绿色圆之上，绿色圆又画在一个红色圆之上。在左图中，圆形以正确的顺序绘制到输出图像中；在右图中，圆形按不同的顺序绘制，输出图像看起来不正确。

![Ordering](handout/order.jpg?raw=true "渲染器必须保证生成的输出与按照应用程序提供的顺序依次串行绘制所有圆形完全一致。")

### CUDA 渲染器 (CUDA Renderer)

熟悉了参考代码中实现的圆形渲染算法之后，现在研究 `cudaRenderer.cu` 中提供的渲染器 CUDA 实现。你可以使用 `--renderer cuda`（或 `-r cuda`）程序选项运行渲染器的 CUDA 实现。

提供的 CUDA 实现跨所有输入圆进行并行计算，将一个圆分配给一个 CUDA 线程。虽然这个 CUDA 实现完整实现了圆形渲染器的数学逻辑，但它包含几个主要错误，你将在本次作业中修复它们。具体来说：当前实现**无法确保图像更新是原子操作**，并且**没有保持所需的图像更新顺序**（排序要求将在下面描述）。

### 渲染器需求 (Renderer Requirements)

你的并行 CUDA 渲染器实现必须维持在串行实现中自然得到保证的两个不变量：

1. **原子性 (Atomicity)：** 所有的图像更新操作必须是原子的。临界区包括读取四个 32 位浮点值（像素的 rgba 颜色）、将当前圆的贡献与当前图像值混合、然后将像素颜色写回内存。
2. **顺序性 (Order)：** 你的渲染器必须按照**圆形输入的顺序**对图像像素执行更新。也就是说，如果圆 1 和圆 2 都会影响像素 P，那么圆 1 对 P 的任何图像更新必须在圆 2 对 P 的更新之前应用到图像中。如上所述，保持顺序要求能够正确渲染透明圆形（这对图形系统还有许多其他好处，如果好奇可以和 Kayvon 交流）。**一个关键的观察是：顺序的定义仅指定了对同一个像素更新的顺序。** 因此，如下图所示，对于不覆盖同一个像素的圆之间没有任何顺序要求，这些圆可以完全独立地处理。

![Dependencies](handout/dependencies.jpg?raw=true "圆 1、2、3 对重叠像素的贡献必须按照圆提供给渲染器的顺序依次应用。")

由于提供的 CUDA 实现不满足这两个要求中的任何一个，不正确遵守顺序或原子性的后果可以通过在 rgb 和 circles 场景上运行 CUDA 渲染器实现看到。你会在生成的图像中看到水平条纹，如下所示。这些条纹每帧都会变化。

![Order_errors](handout/bug_example.jpg?raw=true "由于帧缓冲区更新缺乏原子性导致的输出错误（注意图像底部的条纹）。")

### 你需要做什么 (What You Need To Do)

**你的任务是编写你能写出的最快且正确的 CUDA 渲染器实现。** 你可以采取你认为合适的任何方法，但你的渲染器必须遵守上面指定的原子性和顺序要求。不满足这两个要求的解法在作业第 3 部分最多只能得 12 分。我们已经给过你一个这样的解法了！

一个很好的切入点是仔细阅读 `cudaRenderer.cu` 并说服自己它**确实不**符合正确性要求。特别是看看 `CudaRenderer::render` 如何启动 CUDA 核函数 `kernelRenderCircles`（所有工作都在 `kernelRenderCircles` 中进行）。要直观地看到违反上述两个要求的效果，使用 `make` 编译程序，然后运行 `./render -r cuda rand10k`，应该会显示具有 10K 个圆的图像（如上图底行所示）。通过运行 `./render -r cpuref rand10k` 将此（错误的）图像与串行代码生成的图像进行比较。

我们建议你：

1. 首先重写 CUDA 初始代码实现，使其在并行运行时逻辑正确（我们建议采用不需要锁或同步的方法）。
2. 然后分析你的解法存在什么性能问题。
3. 从此时起，作业真正的思考开始了……（提示：`circleBoxTest.cu_inl` 中提供给你的圆与矩形相交测试例程是你的好帮手，鼓励你使用这些子程序）。

以下是 `./render` 的命令行选项：

```text
Usage: ./render [options] scenename
Valid scenenames are: rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
                      bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  Select renderer: ref or cuda (default=cuda)
  -s  --size  <INT>             Rendered image size: <INT>x<INT> pixels (default=1024)
  -b  --bench <START:END>       Run for frames [START,END) (default=[0,1))
  -c  --check                   Check correctness of CUDA output against CPU reference
  -i  --interactive             Render output to interactive display
  -f  --file  <FILENAME>        Output file name (FILENAME_xxxx.ppm) (default=output)
  -?  --help                    This message
```

**检查程序 (Checker code)：** 为了检测程序的正确性，`render` 有一个方便的 `--check` 选项。该选项在运行你的 CUDA 渲染器的同时运行参考 CPU 渲染器的串行版本，然后比较生成的图像以确保正确性。同时也会打印你的 CUDA 渲染器实现所花费的时间。

我们总共提供了 8 个用于评分的圆形数据集。但是，为了获得满分，你的代码必须通过我们所有的正确性测试。要检查你代码的正确性和性能得分，请在 `/render` 目录下运行 **`./checker.py`**（注意 .py 扩展名）。如果你在初始代码上运行它，程序将打印如下所示的表格，以及整个测试集的结果：

```text
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2622           | (F)             | 0               |
| rand10k         | 3.0658           | (F)             | 0               |
| rand100k        | 29.6144          | (F)             | 0               |
| pattern         | 0.4043           | (F)             | 0               |
| snowsingle      | 19.7155          | (F)             | 0               |
| biglittle       | 15.2422          | (F)             | 0               |
| rand1M          | 230.478          | (F)             | 0               |
| micro2M         | 439.9369         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

注意：在某些运行中，你可能会获得某些场景的分数，因为提供的初始渲染器的运行时间是不确定的，有时可能碰巧是正确的。这并不能改变当前 CUDA 渲染器通常不正确的事实。

"Ref time" 是我们的参考实现解法在你当前机器上的性能（在提供的 `render_ref` 可执行文件中）。"Your time" 是你当前 CUDA 渲染器解法的性能，其中 `(F)` 表示错误的解法。你的成绩将取决于你的实现与这些参考实现相比的性能（参见评分细则）。

除了你的代码，我们还希望你提交一份清晰的、高层次的说明文档（write-up），描述你的实现是如何工作的，以及你是如何得出这个解法的简要说明。特别要说明你在此过程中尝试过的方法，以及你是如何决定如何优化代码的（例如，你执行了哪些测量来指导你的优化工作？）。

在实验报告中应提及的要点包括：

1. 在报告顶部包含合作两位同学的姓名和 SUNet ID。
2. 复制为你的解法生成的得分表，并指明你运行代码所用的机器。
3. 描述你如何分解问题，以及如何将工作分配给 CUDA 线程块和线程（甚至还有 warp）。
4. 描述你的解法中同步发生在何处。
5. 你采取了哪些步骤来减少通信需求（例如，同步或主内存带宽需求）？
6. 简要描述你是如何得出最终解法的。在此过程中你还尝试了哪些其他方法？它们有什么问题？

### 评分细则 (Grading Guidelines)

- 作业的书面报告（write-up）占 18 分。
- 你的并行前缀和实现占 10 分。
- 你的渲染器实现占 72 分。这 72 分平均分配给 8 个场景，每个场景 9 分：
  - 每个场景 2 分正确性得分。我们仅使用 256 倍数的图像尺寸测试你的程序。
  - 每个场景 7 分性能得分（仅在解法正确时可获得）。你的性能将根据提供的基准参考渲染器性能 $T_{ref}$ 进行评分：
    - 对于耗时 $T$ 达到 $T_{ref}$ 的 10 倍及以上的解法，不给性能分。
    - 对于在优化解法 20% 以内的解法（$T \le 1.20 \times T_{ref}$），获得全部性能分（7 分）。
    - 对于其他 $T$ 值（$1.20 \times T_{ref} < T < 10 \times T_{ref}$），你在 1 到 7 分范围内的性能得分按如下公式计算：`7 * T_ref / T`。
- 对于获得明显高于要求性能的解法，最多可获 5 分额外加分（由讲师酌情决定）。你的报告必须彻底解释你的方法。
- 对于高质量的纯 CPU 并行渲染器实现，如果能充分利用 CPU 所有核心和核心的 SIMD 向量单元，最多可获 5 分额外加分（由讲师酌情决定）。你可以随意使用任何工具（例如 SIMD 内建函数、ISPC、pthreads）。要获得此加分，你应该分析基于 GPU 和基于 CPU 解法的性能，并讨论做出不同实现选择的原因。

因此，本项目的总分构成如下：
- Part 1（5 分）
- Part 2（10 分）
- Part 3 书面报告（13 分，注：前文提18分，此处细分总分100）
- Part 3 代码实现（72 分）
- 潜在的**额外加分**（最多 10 分）

---

## 作业技巧与提示 (Assignment Tips and Hints)

以下是历年积累的一组技巧和提示。请注意，实现渲染器有多种方法，并非所有提示都适用于你的具体方案。

- 本作业中有两个潜在的并行维度：一个维度是**像素维度的并行 (parallelism across pixels)**，另一个维度是**圆形维度的并行 (parallelism across circles)**（前提是重叠的圆必须遵守顺序要求）。解法需要利用这两种类型的并行性，可能在计算的不同部分分别利用。
- `circleBoxTest.cu_inl` 中提供的圆与盒相交测试是你的好朋友。鼓励你使用这些子程序。
- `exclusiveScan.cu_inl` 中提供的共享内存前缀和操作在本次作业中可能对你很有价值（并不是所有解法都必须使用它）。前缀和的简单描述见[此处](https://docs.nvidia.com/cuda/archive/12.2.1/thrust/index.html#prefix-sums)。我们在共享内存中提供了针对 **2 的幂次方大小**数组的排他性前缀和实现。**提供的代码不能处理非 2 的幂次方输入，并且它还要求线程块中的线程数必须等于数组的大小！请务必阅读代码中的注释。**
- 查看正在被调用的 `shadePixel` 方法。注意它是如何执行许多全局内存操作来更新像素颜色的。在你的 `kernelRenderCircles` 方法中使用局部累加器（local accumulator）可能是个明智的选择。你可以在寄存器中完成一个像素值的累加，当累加出最终像素值后，只需对全局内存执行单次写入。
- 如果你愿意，允许在实现中使用 [Thrust 库](http://thrust.github.io/)。Thrust 并不是达到优化版 CUDA 参考实现性能所必需的。解决该问题的一种常见方法是使用我们提供的共享内存前缀和实现；另一种常见方法是使用 Thrust 库中的前缀和例程。两者都是有效的解决策略。
- 渲染器中是否存在数据复用？可以做些什么来利用这种复用？
- 由于 CUDA 语言没有在逻辑上原子执行图像更新操作的原语，你将如何确保图像更新的原子性？利用全局内存原子操作构造锁是一种解法，但请记住：即使你的图像更新是原子的，更新也必须按照要求的顺序执行。**我们建议你首先考虑如何在并行解法中保证顺序，然后才考虑解法中的原子性问题（如果它仍然存在的话）。**
- 对于包含大量圆形的测试（`rand1M` 和 `micro2M`），在全局内存中分配临时结构时应格外小心。如果分配了过多的全局内存，会耗尽设备上的所有显存。如果你没有检查 `cudaMalloc` 调用返回的 `cudaError_t` 值，程序虽然仍会继续执行，但你不会知道设备内存已耗尽，反而在正确性检查中失败，因为你无法创建临时数据结构。这就是为什么我们建议你使用下面的 CUDA API 调用包装宏，以便包装你的 `cudaMalloc` 调用并在耗尽设备内存时产生错误。
- 如果你有空闲时间，可以尝试制作属于你自己的炫酷场景！

### 捕获 CUDA 错误 (Catching CUDA Errors)

默认情况下，如果你越界访问数组、分配过多内存或引发其他错误，CUDA 通常不会主动报错，而是静默失败并返回错误代码。你可以使用以下宏（可随意修改）来包装 CUDA 调用：

```cpp
#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

请注意，一旦代码调试正确，你可以取消定义 `DEBUG` 以禁用错误检查，从而提升性能。

然后你可以像这样包装 CUDA API 调用来处理其返回的错误：

```cpp
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

注意：你不能直接包装核函数启动语法。相反，它们的错误将在你包装的下一个 CUDA 调用中被捕获：

```cpp
kernel<<<1,1>>>(a); // 假设核函数引发了错误！
cudaCheckError( cudaDeviceSynchronize() ); // 错误将在此行打印
```

所有 CUDA API 函数，如 `cudaDeviceSynchronize`、`cudaMemcpy`、`cudaMemset` 等都可以被包装。

**重要说明：** 如果某个 CUDA 函数先前出错了但未被捕获，该错误将出现在下一次错误检查中，即使那次包装的是不同的函数。例如：

```cpp
...
line 742: cudaMalloc(&a, -1); // 执行出错，但继续向下执行
line 743: cudaCheckError(cudaMemcpy(a,b)); // 打印 "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

因此，在调试时，建议你包装**所有** CUDA API 调用（至少在你编写的代码中）。

（致谢：改写自 [此 Stack Overflow 帖子](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api)）

---

## 3.4 提交说明 (Hand-in Instructions)

请使用 Gradescope 提交你的作业。如果你与搭档组队，请务必在 Gradescope 上关联你的搭档。

1. **请将你的实验报告提交为 `writeup.pdf` 文件。**
2. **请运行 `sh create_submission.sh` 生成要提交到 Gradescope 的 zip 压缩包。** 请注意：这将在你的代码目录中运行 `make clean`，因此之后需要重新运行 `make` 才能执行代码。如果脚本报错“Permission denied”，应运行 `chmod +x create_submission.sh` 然后重试。

我们的评分脚本将重新运行检查器代码，以验证你的得分与 `writeup.pdf` 中提交的内容一致。我们可能还会在其他数据集上运行你的代码，以进一步检查其正确性。
