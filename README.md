## Add, substract and scale

### Add/Subtract: Do it element-by-element, same dimensions required.

Scale: Multiply everything by the scalar.

1. Addition:

Requirement: Vectors or matrices must have the same dimensions (same number of rows and columns).

How: You add corresponding elements.

[a  b]   +  [e  f]   =  [a+e  b+f]
[c  d]      [g  h]      [c+g  d+h]
Use code with caution.
Same for vectors: [1, 2] + [3, 4] = [4, 6]

2. Subtraction:

Requirement: Same as addition – same dimensions.

How: You subtract corresponding elements.

[a  b]   -  [e  f]   =  [a-e  b-f]
[c  d]      [g  h]      [c-g  d-h]
Use code with caution.
Same for vectors: [1, 2] - [3, 4] = [-2, -2]

3. Scaling (Scalar Multiplication):

Requirement: A single number (the "scalar") and a vector or matrix.

How: Multiply every element of the vector or matrix by the scalar.

k * [a  b]   =  [k*a  k*b]
    [c  d]      [k*c  k*d]

Same for vectors: 2 * [1, 2] = [2, 4]



## Linear combination

- Let u = (u1, ..., uk) ∈ V
k be a list of size k, containing vectors (V is a vector space).

- Let λ = (λ1, ..., λk) ∈ K
k be a list of size k, containing scalars

A linear combination of a set of vectors u1, u2, ..., uk with coefficients λ1, λ2, ..., λk 
is the sum of each vector scaled by its corresponding coefficient:

    λ1 ⋅ u1 + λ2 ⋅ u2 + ⋯ + λk ⋅ uk

Example:

Given:

    u1 = (1, 2, 3),  
    u2 = (0, 10, -100)  

And coefficients:

    λ1 = 10,  
    λ2 = -2  

We multiply each vector by its scalar:

    10 ⋅ u1 = (10 ⋅ 1, 10 ⋅ 2, 10 ⋅ 3) = (10, 20, 30)  

    -2 ⋅ u2 = (-2 ⋅ 0, -2 ⋅ 10, -2 ⋅ (-100)) = (0, -20, 200)  

Adding the results:

    (10, 20, 30) + (0, -20, 200) = (10, 0, 230)


Used for:

- Solving Systems of Linear Equations
- Data Representation and Transformation (Data Science, Machine Learning, Computer Graphics)
    - PCA, Neural networks (weights, bias)


## Linear interpolation

Estimating a value between two known values.
It assumes that the change between the known points is linear (forms a straight line). It's the simplest form of interpolation.

It's essentially a weighted average


## Dot product

AKA Scalar product

Takes two vectors of the same dimension and returns a single scalar (a number)

a = [a₁, a₂, ..., aₙ]

b = [b₁, b₂, ..., bₙ]

a ⋅ b = a₁b₁ + a₂b₂ + ... + aₙbₙ

Machine Learning:

Cosine Similarity: Measuring the similarity between two vectors (e.g., vectors representing documents or images) by calculating the cosine of the angle between them (using the dot product).

Neural Networks: Weights in a neural network can be interpreted as vectors, and the dot product is used in the input layer to calculate the activation of neurons.


## Norm

1. 1-norm (L¹ norm, Taxicab norm, Manhattan norm): ||v||₁
Definition: The 1-norm is the sum of the absolute values of the vector's components.

2. 2-norm (L² norm, Euclidean norm): ||v||₂ or just ||v||
Definition: The 2-norm is the square root of the sum of the squares of the vector's components. This is the "standard" notion of distance in Euclidean space.

3. ∞-norm (L<sup>∞</sup> norm, Infinity norm, Supremum norm, Max norm): ||v||<sub>∞</sub>
Definition: The ∞-norm is the maximum of the absolute values of the vector's components.


## Cosine

Given two vectors **A** and **B**, the cosine of the angle **θ** between them is calculated as:  

    cos(θ) = (A · B) / (|A| |B|)  

Where:  
- **A · B** is the **dot product** of the vectors.  
- **|A|** and **|B|** are the **magnitudes (norms)** of the vectors.  

📌 **The value of cos(θ) is always between -1 and 1**:  
- If **cos(θ) = 1** → The vectors are **parallel** and point in the **same direction**.  
- If **cos(θ) = 0** → The vectors are **perpendicular** (90° angle).  
- If **cos(θ) = -1** → The vectors are parallel but in **opposite directions**.  

Applications of Cosine Between Vectors  

🔹 **Machine Learning**: Used in **"Cosine Similarity"** to measure the similarity between text documents.  
🔹 **Physics**: To calculate forces and angles in mechanics problems.  
🔹 **3D Graphics**: In rendering engines to determine lighting and shadows. 


## Cross product

Given two vectors:

    A = (A_x, A_y, A_z)
    B = (B_x, B_y, B_z)

The **cross product** A × B is calculated as:

    A × B = ((Ay * Bz - Az * By), (Az * Bx - Ax * Bz), (Ax * By - Ay * Bx))

- The result is always **perpendicular** to both A and B.


## Matrix Multiplication


Given two matrices:

    A = [a11  a12  a13]      B = [b11  b12]
        [a21  a22  a23]          [b21  b22]
        [a31  a32  a33]          [b31  b32]

Matrix A is 3x3 (3 rows, 3 columns), and matrix B is 3x2 (3 rows, 2 columns).

To multiply these matrices, take the **dot product** of rows from A with columns from B.

The result is a new matrix C:

    C = A * B = [c11  c12]
                [c21  c22]
                [c31  c32]

Where each element in C is calculated by:

    c11 = (a11 * b11) + (a12 * b21) + (a13 * b31)
    c12 = (a11 * b12) + (a12 * b22) + (a13 * b32)
    c21 = (a21 * b11) + (a22 * b21) + (a23 * b31)
    c22 = (a21 * b12) + (a22 * b22) + (a23 * b32)
    c31 = (a31 * b11) + (a32 * b21) + (a33 * b31)
    c32 = (a31 * b12) + (a32 * b22) + (a33 * b32)

So the number of columns in the first matrix (A) must be equal to the number of rows in the second matrix (B).

Applications of Matrix Multiplication

- **Computer Graphics**: Used to transform images and perform rotations, scaling, and translations.
- **Data Science**: Applied in machine learning models, like neural networks and linear regression.
- **Physics and Engineering**: Helps in solving systems of linear equations and transformations.


## Trace of a Matrix

The **trace** of a matrix is the sum of the elements on its **main diagonal** (the diagonal that runs from the top-left corner to the bottom-right corner). The trace is only defined for **square matrices** (matrices with the same number of rows and columns).


Given a square matrix:

    A = [a11  a12  a13]
        [a21  a22  a23]
        [a31  a32  a33]

The trace of matrix A, denoted as **Tr(A)**, is calculated by summing the elements on the diagonal:

    Tr(A) = a11 + a22 + a33

Properties of the Trace

- The trace is only defined for **square matrices** (same number of rows and columns).
- The trace is **invariant** under a **change of basis**, meaning it remains the same if the matrix is transformed in a different coordinate system.
- The trace is **linear**, meaning

Applications of the Trace

- **Linear Algebra**: Used in matrix theory and to compute invariants of matrices.
- **Machine Learning**: Appears in optimization problems like the trace norm used in matrix completion.
- **Physics**: The trace is used in quantum mechanics and in calculating the properties of certain physical systems.



## Transpose of a Matrix

The **transpose** of a matrix is a new matrix obtained by **flipping** the matrix over its diagonal. This means the rows of the original matrix become the columns of the new matrix, and the columns become the rows.

Mathematical Definition

Given a matrix:

    A = [a11  a12  a13]
        [a21  a22  a23]
        [a31  a32  a33]

The transpose of matrix A, denoted as **A^T**, is:

    A^T = [a11  a21  a31]
          [a12  a22  a32]
          [a13  a23  a33]


Properties of the Transpose

- **(A^T)^T = A**: The transpose of the transpose of a matrix is the original matrix.
- **(A + B)^T = A^T + B^T**: The transpose of the sum of two matrices is the sum of their transposes.
- **(cA)^T = cA^T**: The transpose of a scalar multiplied by a matrix is the scalar multiplied by the transpose of the matrix.
- **(AB)^T = B^T A^T**: The transpose of a product of two matrices is the product of their transposes in reverse order.

Applications of the Transpose

- **Linear Algebra**: Used in many matrix operations, including matrix multiplication and solving systems of equations.
- **Machine Learning**: Helps in operations like backpropagation in neural networks.
- **Computer Graphics**: Transposes are used to change between different coordinate systems or to rotate objects.



## Row Echelon Form (REF)

**Row Echelon Form (REF)** is a specific arrangement of a matrix used to simplify the process of solving systems of linear equations. It’s a key concept in **linear algebra**, especially when using **Gaussian elimination** to solve systems.

A matrix is in **Row Echelon Form (REF)** if it satisfies the following conditions:

1. **All rows with non-zero elements** are above any rows with only zeros.
2. The **leading entry (pivot) in each non-zero row** is **1**, and it is the only non-zero entry in its column.
3. The **leading entry** of each subsequent row is **to the right** of the leading entry of the row above it.

Consider the matrix:

    A = [1  2  3  |  9]
        [4  5  6  | 10]
        [7  8  9  | 11]

To convert it into Row Echelon Form, we perform row operations such as row swaps, scaling rows, and adding multiples of one row to another. After performing these steps, the matrix looks like this:

    REF(A) = [1  2  3  |  9]
             [0  1  2  |  1]
             [0  0  1  |  0]

This matrix is now in Row Echelon Form because:

- All non-zero rows are above any rows with all zeros.
- The leading entry (pivot) of each row is 1 and is the only non-zero element in its column.
- The pivots are shifted to the right as you move down the rows.

Steps to Convert a Matrix to Row Echelon Form

1. **Find the first non-zero element** in the first column. This becomes the pivot.
2. **Create zeros below the pivot** by using row operations (e.g., subtracting multiples of the pivot row from the rows below).
3. **Move to the next column** and repeat the process for each subsequent row.
4. Ensure that **each pivot** is the only non-zero element in its column.


Applications of Row Echelon Form

- **Solving Linear Systems**: REF is the first step in Gaussian elimination, which is used to solve systems of linear equations.
- **Finding Inverses**: REF is useful when applying row reduction to find the inverse of a matrix.
- **Linear Independence**: REF helps determine if a set of vectors is linearly independent or dependent by examining the number of pivots.


## Determinant of a Matrix

### If determinant == 0
- The matrix is not invertible.
- The rows/columns are linearly dependent.
- The transformation represented by the matrix collapses space to a lower dimension.
- The associated system of linear equations does not have a unique solution.
- There is at least one zero eigenvalue.

### Geometric representation
The determinant of a matrix, in the context of a linear transformation, tells you how much the transformation scales areas/volumes/hypervolumes (the absolute value of the determinant) and whether it preserves or reverses orientation (the sign of the determinant). A determinant of 0 indicates a collapse to a lower dimension. It's a powerful tool for understanding the geometric effects of linear transformations.


The **determinant** of a matrix is a scalar value that can be computed from the elements of a square matrix. It provides important information about the matrix, such as whether it is invertible and how it scales space.

For a **square matrix** (same number of rows and columns), the **determinant** is a value that can be calculated based on the elements of the matrix.

- For a 2x2 Matrix:

Given a matrix:

    A = [a  b]
        [c  d]

The determinant is calculated as:

    det(A) = (a * d) - (b * c)

- For a 3x3 Matrix:

Given a matrix:

    A = [a11  a12  a13]
        [a21  a22  a23]
        [a31  a32  a33]

The determinant is:

    det(A) = a11 * ((a22 * a33) - (a23 * a32)) - a12 * ((a21 * a33) - (a23 * a31)) + a13 * ((a21 * a32) - (a22 * a31))

Properties of the Determinant

- **Invertibility**: A matrix is **invertible** (has an inverse) if and only if its determinant is **non-zero**. If det(A) = 0, the matrix is **singular** and has no inverse.
- **Scaling**: The determinant can be used to scale areas or volumes when transforming geometric shapes with a matrix.
- **Row Operations**:
  - If you swap two rows, the determinant changes sign.
  - If you multiply a row by a scalar, the determinant is multiplied by that scalar.
  - If you add a multiple of one row to another, the determinant does not change.

Applications of the Determinant

- **Solving Systems of Equations**: The determinant is used in Cramer's rule to solve linear systems of equations.
- **Matrix Inversion**: Determines if a matrix is invertible.
- **Geometric Interpretation**: In 2D or 3D space, the determinant can represent the area (in 2D) or volume (in 3D) of a shape transformed by a matrix.


## Inverse of a Matrix

The **inverse** of a matrix is a matrix that, when multiplied by the original matrix, results in the **identity matrix**. The inverse is only defined for **square matrices** (matrices with the same number of rows and columns), and not all square matrices have an inverse.

Mathematical Definition

Given a square matrix **A**, the **inverse** of matrix A, denoted as **A⁻¹**, satisfies the following condition:

    A * A⁻¹ = A⁻¹ * A = I

Where **I** is the **identity matrix** (a matrix with 1s on the diagonal and 0s elsewhere).

- A matrix **A** has an inverse if and only if its **determinant is non-zero** (det(A) ≠ 0).
- If **det(A) = 0**, the matrix is called **singular**, and it does not have an inverse.

How to Find the Inverse of a 2x2 Matrix

Given a 2x2 matrix:

    A = [a  b]
        [c  d]

The inverse of A is:

    A⁻¹ = (1 / det(A)) * [d  -b]
                         [-c  a]

Where det(A) = (a * d) - (b * c).

How to Find the Inverse of a 3x3 Matrix

For a 3x3 matrix, the inverse is more complex to compute. It involves finding the **adjugate matrix** and dividing it by the determinant of the original matrix.

Given a 3x3 matrix **A**:

    A = [a11  a12  a13]
        [a21  a22  a23]
        [a31  a32  a33]

The inverse of A, A⁻¹, can be found using the formula:

    A⁻¹ = (1 / det(A)) * adj(A)

Where **adj(A)** is the adjugate matrix of A, which is the transpose of the cofactor matrix.


Properties of the Inverse

- **(A⁻¹)⁻¹ = A**: The inverse of the inverse of a matrix is the original matrix.
- **(A * B)⁻¹ = A⁻¹ * B⁻¹**: The inverse of the product of two matrices is the product of their inverses in reverse order.
- **(cA)⁻¹ = (1/c) * A⁻¹**: The inverse of a scalar multiple of a matrix is the scalar reciprocal times the inverse of the matrix.
- **(A + B)⁻¹ ≠ A⁻¹ + B⁻¹**: The inverse of a sum is not the sum of the inverses.

Applications of the Inverse

- **Solving Systems of Equations**: The inverse is used to solve linear systems using matrix equations, such as A * X = B, where X = A⁻¹ * B.
- **Linear Transformations**: The inverse matrix is used to reverse transformations in fields like computer graphics and physics.
- **Optimization**: In optimization problems, especially in machine learning, matrix inverses are used to solve for parameters efficiently (e.g., in linear regression).



## Rank of a Matrix

The **rank** of a matrix is the number of **linearly independent rows or columns** in the matrix. It gives an idea of the matrix's dimension in terms of its row or column space, and it's important in determining whether a matrix can be inverted or how it transforms space.

Mathematical Definition

The rank of a matrix is the dimension of the **row space** or **column space** of the matrix. It represents the maximum number of linearly independent rows or columns in the matrix.

- **Row rank**: The number of linearly independent rows in the matrix.
- **Column rank**: The number of linearly independent columns in the matrix.

For any matrix, the **row rank** and **column rank** are always the same. This is known as the **rank theorem**.

How to Calculate the Rank of a Matrix

The rank of a matrix can be determined by transforming the matrix into **row echelon form** (REF) or **reduced row echelon form** (RREF), and then counting the number of non-zero rows (or pivot positions).

Example:

Consider the matrix:

    A = [1  2  3]
        [4  5  6]
        [7  8  9]

To find the rank:

1. **Transform to Row Echelon Form (REF)**:

    After row reductions, we get:

    REF(A) = [1  2  3]
             [0  -3  -6]
             [0  0  0]

2. **Count the number of non-zero rows**: The matrix has 2 non-zero rows, so the rank of this matrix is **2**.

Properties of the Rank

- **Max Rank**: For an **m x n** matrix, the rank is at most **min(m, n)**.
- **Full Rank**: A matrix is said to be of **full rank** if its rank is equal to the smaller of the number of rows or columns (i.e., **rank(A) = min(m, n)**).
- **Rank and Invertibility**: If a square matrix (n x n) has full rank (i.e., its rank is n), it is **invertible**. If its rank is less than n, it is **singular** and does not have an inverse.
- **Rank and Linear Independence**: The rank represents the number of linearly independent rows or columns. If the rank of a matrix is less than the number of rows or columns, there are dependent rows or columns.

4. Applications of the Rank

- **Solving Systems of Linear Equations**: The rank helps determine whether a system of equations has a unique solution, no solution, or infinitely many solutions.
- **Linear Dependence and Independence**: The rank can be used to determine whether a set of vectors is linearly independent.
- **Matrix Inversion**: A square matrix is invertible if and only if it has full rank.
- **Data Compression**: In data science and machine learning, the rank of a matrix can be used to perform techniques like **Principal Component Analysis (PCA)** for dimensionality reduction.
- **Control Theory**: In engineering, the rank is used to determine the controllability and observability of systems.
