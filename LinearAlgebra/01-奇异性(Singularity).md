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
if det_A == 0
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
print('矩阵A是奇异的' if abs(det_A) < 1e-10 else '矩阵A不是奇异的')
```

---

### **2. 通过矩阵的秩（Rank）判断**

#### **秩的定义**
矩阵的**秩**是一个衡量矩阵"有效维度"的指标：
1. **线性无关向量的最大数量**：矩阵中线性无关的行向量或列向量的最大个数
2. **等价定义**：
   - **行秩**：线性无关的行向量的最大数量
   - **列秩**：线性无关的列向量的最大数量
   - 根据**秩的等式定理**，行秩 = 列秩，因此可以简单称为矩阵的秩

#### **计算矩阵的秩**
1. **初等行变换法**：
   - 将矩阵化为**行阶梯形矩阵**
   - 非零行的数量即为矩阵的秩
   - 具体步骤：
     1. 通过初等行变换将矩阵转化为行阶梯形
     2. 继续变换得到行最简形
     3. 计算非零行数量

2. **子式法**：
   - 检查各阶子式是否为零
   - 最高阶非零子式的阶数即为矩阵的秩

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
print('矩阵B是奇异的' if rank_B < n else '矩阵B不是奇异的')
```

---

### **3. 通过逆矩阵（Inverse Matrix）判断**
如果矩阵 **没有逆矩阵**（即 \( A^{-1} \) 不存在），则 \( A \) 是奇异的。逆矩阵的计算方法：
\[
A^{-1} = \frac{1}{\det(A)} \text{Adj}(A)
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
求特征值：
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
print('矩阵C是奇异的' if np.any(np.abs(eig_vals) < 1e-10) else '矩阵C不是奇异的')
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

### **总结**
判断矩阵 \( A \) 是否**奇异**的方法：
1. **计算行列式** \( \det(A) \)，如果 \( \det(A) = 0 \)，则 \( A \) 是奇异的。
2. **计算矩阵的秩** \( \text{Rank}(A) \)，如果小于矩阵阶数，则 \( A \) 是奇异的。
3. **检查是否可逆**，如果矩阵无逆，则是奇异的。
4. **计算特征值**，如果某个特征值是 0，则矩阵是奇异的。
