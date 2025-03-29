# 01-线性方程组的 **奇异性(Singularity)**

在**线性方程组**或**矩阵**的背景下，**奇异性（Singularity）** 通常指矩阵不可逆，即行列式为零。计算奇异性的方法如下：

---

### **1. 计算行列式（Determinant）**
对于一个 \( n \times n \) 的矩阵 \( A \)，如果 **行列式** \( \det(A) = 0 \)，则 \( A \) 是**奇异矩阵（Singular Matrix）**，否则是**非奇异矩阵（Non-Singular Matrix）**。

#### **示例**
对于矩阵：
\[
A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}
\]
计算行列式：
\[
\det(A) = 1 \times 6 - 2 \times 3 = 6 - 6 = 0
\]
因此 \( A \) 是**奇异矩阵**。

#### **代码实现**
```matlab
% MATLAB代码
A = [1 2; 3 6];
det_A = det(A)  % 计算行列式
if abs(det_A) < 1e-10  % 使用阈值判断，而不是精确的0
    disp('矩阵A是奇异的')
else
    disp('矩阵A不是奇异的')
end
```

```python
# Python代码
import numpy as np

A = np.array([[1, 2], [3, 6]])
det_A = np.linalg.det(A)  # 计算行列式
print(f'行列式值: {det_A}')
if abs(det_A) < 1e-10:
    print('矩阵A是奇异的')
else:
    print('矩阵A不是奇异的')

# 注意：在数值计算中，由于浮点数精度限制，不应直接使用 == 0 判断
# 例如：0.1 + 0.2 = 0.30000000000000004
# 因此使用一个小的阈值(如1e-10)来判断数字是否足够接近0
# 这样可以避免因为舍入误差导致的错误判断
```

---

### **2. 通过矩阵的秩（Rank）判断**

#### **线性无关的概念**
两个或多个向量的**线性无关**是指：一个向量不能由其他向量的线性组合（缩放和加减）表示。例如：
- 向量 \( v_1 = [1,0] \) 和 \( v_2 = [0,1] \) 是线性无关的，因为无法通过缩放其中一个向量得到另一个
- 向量 \( v_1 = [1,2] \) 和 \( v_2 = [2,4] \) 是线性相关的，因为 \( v_2 = 2v_1 \)

用数学语言表达：若向量组 \( v_1, v_2, ..., v_n \) 满足：
\[
c_1v_1 + c_2v_2 + ... + c_nv_n = 0
\]
当且仅当所有系数 \( c_i = 0 \) 时等式成立，则称这组向量**线性无关**。

#### **秩的定义**
矩阵的**秩**是一个衡量矩阵"有效维度"的指标：
1. **线性无关向量的最大数量**：矩阵中线性无关的行向量或列向量的最大个数
2. **等价定义**：
   - **行秩**：线性无关的行向量的最大数量
   - **列秩**：线性无关的列向量的最大数量
   - **秩的等式定理**：对于任意矩阵，其行秩等于列秩
     - 这个定理说明了矩阵的一个重要性质：行向量的线性相关性与列向量的线性相关性是等价的
     - 例如：对于矩阵 \( \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \)
       * 行向量是 [1,2] 和 [2,4]，第二行是第一行的2倍，所以行秩是1
       * 列向量是 [1,2]ᵀ 和 [2,4]ᵀ，第二列是第一列的2倍，所以列秩也是1
     - 这个性质使我们可以简单地用"秩"来表示矩阵的线性无关向量的最大数量，而不需要区分行秩和列秩

#### **计算矩阵的秩**
1. **初等行变换法**：
   - **行阶梯形矩阵（Row Echelon Form）的定义**：
     1. 零行（全为0的行）都在矩阵底部
     2. 每个非零行的首个非零元素（主元）左边的所有元素都是0
     3. 每个主元所在的列，其下方的元素都是0
     4. 每个主元右边可以有任意数字
   
   例如，以下是一个行阶梯形矩阵：
   \[
   \begin{bmatrix} 
   1 & 2 & 3 & 4 \\
   0 & 1 & 2 & 3 \\
   0 & 0 & 1 & 2 \\
   0 & 0 & 0 & 0
   \end{bmatrix}
   \]
   
   **初等行变换的三种基本操作**：
   1. 交换两行的位置
   2. 用一个非零数乘以某一行
   3. 将某一行的倍数加到另一行上

   - 将矩阵化为**行阶梯形矩阵**
   - 非零行的数量即为矩阵的秩
   - 具体步骤：
     1. 通过初等行变换将矩阵转化为行阶梯形
     2. 继续变换得到行最简形
     3. 计算非零行数量
   
   **示例**：
   对于矩阵：
   \[
   A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 3 & 4 \end{bmatrix}
   \]
   通过初等行变换：
   \[
   \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 3 & 4 \end{bmatrix} 
   \xrightarrow{R_2-2R_1} \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 1 & 3 & 4 \end{bmatrix}
   \xrightarrow{R_3-R_1} \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & 1 & 1 \end{bmatrix}
   \]
   最终得到两个非零行，因此矩阵的秩为2。

   **代码实现**：
   ```matlab
   % MATLAB代码
   A = [1 2 3; 2 4 6; 1 3 4];
   % 使用rref获取行最简形矩阵
   [R, jb] = rref(A);
   disp('行最简形矩阵:')
   disp(R)
   disp(['矩阵的秩: ', num2str(length(jb))])
   ```

   ```python
   # Python代码
   import numpy as np
   from scipy import linalg

   A = np.array([[1, 2, 3], [2, 4, 6], [1, 3, 4]])
   # 使用SVD分解计算秩
   rank_A = np.linalg.matrix_rank(A)
   print(f'矩阵的秩: {rank_A}')
   
   # 使用高斯消元（可选）
   def gauss_rank(A):
       R = linalg.qr(A, mode='r')[0]  # QR分解的R部分
       tol = 1e-10  # 容差
       return sum(np.abs(np.diag(R)) > tol)
   
   print(f'通过高斯消元计算的秩: {gauss_rank(A)}')
   ```

2. **子式法**：
   - 检查各阶子式是否为零
   - 最高阶非零子式的阶数即为矩阵的秩
   
   **示例**：
   对于矩阵：
   \[
   A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
   \]
   计算各阶子式：
   - 1阶子式：|1| ≠ 0，|2| ≠ 0，|2| ≠ 0，|4| ≠ 0
   - 2阶子式：\( \begin{vmatrix} 1 & 2 \\ 2 & 4 \end{vmatrix} = 0 \)
   因此最高非零阶数为1，矩阵的秩为1。

   **代码实现**：
   ```matlab
   % MATLAB代码：计算子式
   A = [1 2; 2 4];
   % 计算所有1阶子式
   minors1 = diag(A);
   % 计算2阶子式（此例中就是行列式）
   minor2 = det(A);
   disp('1阶子式:')
   disp(minors1)
   disp('2阶子式:')
   disp(minor2)
   ```

   ```python
   # Python代码：计算子式
   import numpy as np
   from itertools import combinations

   def compute_minors(A, order):
       n = A.shape[0]
       minors = []
       # 获取所有可能的行和列组合
       indices = list(combinations(range(n), order))
       for rows in indices:
           for cols in indices:
               # 提取子矩阵
               sub_matrix = A[np.ix_(rows, cols)]
               minors.append(np.linalg.det(sub_matrix))
       return np.array(minors)

   A = np.array([[1, 2], [2, 4]])
   minors1 = compute_minors(A, 1)  # 1阶子式
   minors2 = compute_minors(A, 2)  # 2阶子式
   print('1阶子式:', minors1)
   print('2阶子式:', minors2)
   ```

#### **奇异性判断**
如果矩阵的秩小于矩阵的大小（即 \(\text{Rank}(A) < n\)），则矩阵是**奇异的**。

#### **示例**
矩阵：
\[
B = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
\]
计算矩阵的秩：
\[
\text{Rank}(B) = 2 < 3
\]
因此，矩阵 \( B \) **是奇异矩阵**。

#### **代码实现**
```matlab
% MATLAB代码
B = [1 2 3; 4 5 6; 7 8 9];
rank_B = rank(B)  % 计算矩阵的秩
n = size(B, 1);   % 矩阵的大小
if rank_B < n
    disp('矩阵B是奇异的')
else
    disp('矩阵B不是奇异的')
end
```

```python
# Python代码
import numpy as np

B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank_B = np.linalg.matrix_rank(B)  # 计算矩阵的秩
n = B.shape[0]  # 矩阵的大小
print(f'矩阵的秩: {rank_B}')
if rank_B < n:
    print('矩阵B是奇异的')
else:
    print('矩阵B不是奇异的')
```

---

### **3. 通过逆矩阵（Inverse Matrix）判断**
如果矩阵 **没有逆矩阵**（即 \( A^{-1} \) 不存在），则 \( A \) 是奇异的。逆矩阵的计算方法：
\[
A^{-1} = \frac{1}{\det(A)} \text{Adj}(A)
\]

#### **伴随矩阵（Adjugate Matrix）Adj(A)的计算**
伴随矩阵是由余子式矩阵的转置构成的，计算步骤：

1. **余子式（Cofactor）**：
   对于矩阵 \( A \) 的第 i 行、第 j 列的元素 \( a_{ij} \)，其余子式 \( C_{ij} \) 是：
   - 删除第 i 行和第 j 列后得到的子矩阵的行列式
   - 再乘以 \( (-1)^{i+j} \)

2. **代数余子式矩阵**：
   由所有余子式组成的矩阵，记作 \( C \)

3. **伴随矩阵**：
   代数余子式矩阵的转置，即 \( \text{Adj}(A) = C^T \)

**示例**：
对于2×2矩阵：
\[
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
\]
其伴随矩阵为：
\[
\text{Adj}(A) = \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
\]

如果 \( \det(A) = 0 \)，则分母为 0，矩阵 \( A \) **没有逆矩阵**，说明它是奇singular的。

#### **代码实现**
```matlab
% MATLAB代码
A = [1 2; 3 6];
try
    inv_A = inv(A);  % 尝试计算逆矩阵
    disp('矩阵A可逆（非奇异）')
    disp('逆矩阵为:')
    disp(inv_A)
catch
    disp('矩阵A不可逆（奇异）')
end
```

```python
# Python代码
import numpy as np

A = np.array([[1, 2], [3, 6]])
try:
    inv_A = np.linalg.inv(A)  # 尝试计算逆矩阵
    print('矩阵A可逆（非奇异）')
    print('逆矩阵为:\n', inv_A)
except np.linalg.LinAlgError:
    print('矩阵A不可逆（奇异）')
```

---

### **4. 通过特征值（Eigenvalues）判断**
如果矩阵的 **特征值（Eigenvalues）** 中包含 0，则矩阵是**奇异的**。

#### **示例**
对于矩阵：
\[
C = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}
\]
求特征值，其中：
- λ（lambda）是**特征值**：我们要求的未知数
- I 是**单位矩阵**：对角线是1，其他位置是0的矩阵，即 \( \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \)
- C-λI 表示：将矩阵C的对角线元素都减去λ

因此：
\[
C - \lambda I = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix} - \lambda\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 4-\lambda & 2 \\ 2 & 1-\lambda \end{bmatrix}
\]

计算行列式：
\[
\det(C - \lambda I) = \begin{vmatrix} 4-\lambda & 2 \\ 2 & 1-\lambda \end{vmatrix}
\]
\[
(4-\lambda)(1-\lambda) - (2 \times 2) = 4 - \lambda - 4\lambda + \lambda^2 - 4 = \lambda^2 - 5\lambda
\]
\[
\lambda(\lambda - 5) = 0
\]
解得特征值 \( \lambda_1 = 0, \lambda_2 = 5 \)，由于一个特征值是 0，因此矩阵是**奇异的**。

#### **代码实现**
```matlab
% MATLAB代码
C = [4 2; 2 1];
eig_vals = eig(C);  % 计算特征值
disp('特征值:')
disp(eig_vals)
if any(abs(eig_vals) < 1e-10)  % 检查是否有接近0的特征值
    disp('矩阵C是奇异的')
else
    disp('矩阵C不是奇异的')
end
```

```python
# Python代码
import numpy as np

C = np.array([[4, 2], [2, 1]])
eig_vals = np.linalg.eigvals(C)  # 计算特征值
print('特征值:', eig_vals)
if np.any(np.abs(eig_vals) < 1e-10):
    print('矩阵C是奇异的')
else:
    print('矩阵C不是奇异的')
```

---

### **5. 奇异性对线性方程组求解的影响**

当我们求解线性方程组 \( Ax = b \) 时，矩阵 \( A \) 的奇异性会直接影响求解过程：

1. **无解或无穷多解**
   - 当矩阵 \( A \) 是奇异矩阵时，线性方程组要么**无解**，要么有**无穷多解**
   - 绝不会有唯一解

2. **原因分析**
   - 从几何角度：奇异矩阵将空间"压缩"到更低维度
     - 在二维平面中的直观理解：
       - 两个方程对应平面上的两条直线
       - 当矩阵奇异时，这两条直线要么平行（无解），要么重合（无穷多解）
       - 这是因为奇异性使得两个方程成为线性相关（一个方程可由另一个方程缩放得到）
     - 更高维空间同理，空间被"压缩"到更低维度
   - 从代数角度：
     - \( \det(A) = 0 \) 意味着矩阵的列向量**线性相关**
     - 这导致方程组中的某些方程可以由其他方程线性组合得到
     - 使得方程组信息冗余或矛盾

3. **实际应用中的启示**
   - 在数值计算中，应避免使用接近奇异的矩阵（病态矩阵）
   - 如果遇到奇异矩阵，可以考虑：
     - 使用广义逆（如伪逆）求解
     - 采用正则化方法
     - 重新建模问题以避免奇异性

#### **代码实现**
```matlab
% MATLAB代码：展示奇异矩阵导致的方程组求解情况
% 例1：无解情况（平行线）
A1 = [1 2; 2 4];  % 奇异矩阵（第二行是第一行的2倍）
b1 = [1; 3];      % 不相容的右端向量
try
    x1 = A1\b1;   % 尝试求解 A1x = b1
catch
    disp('方程组无解（直线平行）')
end

% 例2：无穷多解情况（重合线）
A2 = [1 2; 2 4];  % 同样的奇异矩阵
b2 = [1; 2];      % 相容的右端向量（第二个元素是第一个的2倍）
x2 = A2\b2;       % 尝试求解 A2x = b2
disp('一个可能的解：')
disp(x2)
disp('但存在无穷多解')
```

```python
# Python代码：展示奇异矩阵导致的方程组求解情况
import numpy as np

# 例1：无解情况（平行线）
A1 = np.array([[1, 2], [2, 4]])  # 奇异矩阵
b1 = np.array([1, 3])            # 不相容的右端向量
try:
    x1 = np.linalg.solve(A1, b1)
except np.linalg.LinAlgError:
    print('方程组无解（直线平行）')

# 例2：无穷多解情况（重合线）
A2 = np.array([[1, 2], [2, 4]])  # 同样的奇异矩阵
b2 = np.array([1, 2])            # 相容的右端向量
try:
    # 使用伪逆找到一个可能的解
    x2 = np.linalg.pinv(A2) @ b2
    print('一个可能的解:', x2)
    print('但存在无穷多解')
except np.linalg.LinAlgError:
    print('计算错误')
```

---

### 6. **总结与比较**
判断矩阵 \( A \) 是否**奇异**有多种方法，每种方法都有其特点和适用场景：

#### **1. 计算行列式**
- **优点**：
  - 对于小型矩阵（2×2, 3×3）计算简单直观
  - 结果明确，没有歧义
- **缺点**：
  - 大型矩阵计算复杂且容易出错
  - 数值计算中可能出现精度问题
- **适用场景**：小型矩阵的快速判断

#### **2. 计算矩阵的秩**
- **优点**：
  - 提供了矩阵的更多结构信息
  - 可以判断不是方阵的矩阵
  - 通过高斯消元可以稳定计算
- **缺点**：
  - 计算过程相对复杂
  - 需要完整的行变换过程
- **适用场景**：需要了解矩阵结构或处理非方阵的情况

#### **3. 检查是否可逆**
- **优点**：
  - 实现简单，大多数编程语言都有内置函数
  - 直接判断是否有解
- **缺点**：
  - 不提供额外的矩阵信息
  - 计算逆矩阵可能不稳定
- **适用场景**：快速程序判断，尤其是使用现成的库函数时

#### **4. 计算特征值**
- **优点**：
  - 提供矩阵的重要特性信息
  - 与矩阵的其他性质有深入联系
- **缺点**：
  - 计算复杂度高
  - 对于大矩阵计算困难
- **适用场景**：需要同时分析矩阵其他特性（如对角化）时

#### **使用建议**
1. **实际编程中**：
   - 小矩阵（n ≤ 3）：使用行列式判断
   - 大矩阵：使用库函数检查可逆性
   - 需要详细分析：计算秩或特征值

2. **数值计算中**：
   - 避免直接判断是否等于0
   - 使用合适的阈值（如1e-10）
   - 优先选择稳定的算法（如QR分解）

3. **理论分析中**：
   - 行列式法适合手工计算和证明
   - 秩的方法适合理解矩阵结构
   - 特征值方法适合研究矩阵性质
