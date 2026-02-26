"""
Matrix Library - A NumPy-like library for matrix operations using Python lists
================================================================================
This library provides a comprehensive set of matrix operations similar to NumPy,
but using pure Python lists as the internal data structure.
"""

class Matrix:
    """
    A class for creating and manipulating matrices using Python lists.
    
    Attributes:
        data (list): 2D list representing the matrix
        rows (int): Number of rows
        cols (int): Number of columns
    """
    
    def __init__(self, data):
        """
        Initialize a Matrix from a 2D list.
        
        Args:
            data: A 2D list (list of lists) representing the matrix
            
        Raises:
            ValueError: If the input is not a valid 2D list or has inconsistent row lengths
        """
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Input must be a non-empty 2D list")
        
        if not all(isinstance(row, list) for row in data):
            raise ValueError("Each row must be a list")
        
        # Check for consistent row lengths
        row_lengths = [len(row) for row in data]
        if len(set(row_lengths)) > 1:
            raise ValueError("All rows must have the same length")
        
        if row_lengths[0] == 0:
            raise ValueError("Rows cannot be empty")
        
        self.data = [row[:] for row in data]  # Deep copy
        self.rows = len(data)
        self.cols = len(data[0])
    
    # ==================== Static Factory Methods ====================
    
    @staticmethod
    def zeros(rows, cols):
        """Create a matrix of zeros."""
        return Matrix([[0 for _ in range(cols)] for _ in range(rows)])
    
    @staticmethod
    def ones(rows, cols):
        """Create a matrix of ones."""
        return Matrix([[1 for _ in range(cols)] for _ in range(rows)])
    
    @staticmethod
    def identity(n):
        """Create an n×n identity matrix."""
        if n <= 0:
            raise ValueError("Size must be positive")
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    @staticmethod
    def random(rows, cols, min_val=0, max_val=1):
        """Create a matrix with random values."""
        import random
        return Matrix([[random.uniform(min_val, max_val) for _ in range(cols)] 
                      for _ in range(rows)])
    
    @staticmethod
    def diagonal(values):
        """Create a diagonal matrix from a list of values."""
        n = len(values)
        return Matrix([[values[i] if i == j else 0 for j in range(n)] for i in range(n)])
    
    # ==================== Basic Properties ====================
    
    @property
    def shape(self):
        """Return the shape of the matrix as a tuple (rows, cols)."""
        return (self.rows, self.cols)
    
    def T(self):
        """Return the transpose of the matrix."""
        return Matrix([[self.data[j][i] for j in range(self.rows)] 
                      for i in range(self.cols)])
    
    @property
    def transpose(self):
        """Return the transpose of the matrix."""
        return self.T()
    
    def flatten(self):
        """Flatten the matrix into a 1D list."""
        return [element for row in self.data for element in row]
    
    def reshape(self, new_rows, new_cols):
        """Reshape the matrix to new dimensions."""
        flat = self.flatten()
        if new_rows * new_cols != len(flat):
            raise ValueError(f"Cannot reshape {self.rows}x{self.cols} to {new_rows}x{new_cols}")
        return Matrix([flat[i*new_cols:(i+1)*new_cols] for i in range(new_rows)])
    
    # ==================== Arithmetic Operations ====================
    
    def __add__(self, other):
        """Add two matrices or a matrix and a scalar."""
        if isinstance(other, Matrix):
            return self._matrix_addition(other)
        elif isinstance(other, (int, float)):
            return self._scalar_addition(other)
        else:
            raise TypeError(f"Unsupported operand type for +: 'Matrix' and '{type(other)}'")
    
    def __radd__(self, other):
        """Reverse add (scalar + matrix)."""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtract from matrix."""
        if isinstance(other, Matrix):
            return self._matrix_subtraction(other)
        elif isinstance(other, (int, float)):
            return self._scalar_subtraction(other)
        else:
            raise TypeError(f"Unsupported operand type for -: 'Matrix' and '{type(other)}'")
    
    def __rsub__(self, other):
        """Reverse subtract (scalar - matrix)."""
        if isinstance(other, (int, float)):
            return self._reverse_scalar_subtraction(other)
        else:
            raise TypeError(f"Unsupported operand type for -: '{type(other)}' and 'Matrix'")
    
    def __mul__(self, other):
        """Multiply matrix by scalar or matrix."""
        if isinstance(other, Matrix):
            return self._matrix_multiplication(other)
        elif isinstance(other, (int, float)):
            return self._scalar_multiplication(other)
        else:
            raise TypeError(f"Unsupported operand type for *: 'Matrix' and '{type(other)}'")
    
    def __rmul__(self, other):
        """Reverse multiply (scalar * matrix)."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Divide matrix by scalar."""
        if isinstance(other, (int, float)):
            return self._scalar_division(other)
        else:
            raise TypeError(f"Unsupported operand type for /: 'Matrix' and '{type(other)}'")
    
    def __neg__(self):
        """Negate the matrix."""
        return self._scalar_multiplication(-1)
    
    def __pow__(self, exponent):
        """Raise matrix to a power (only for square matrices and integer exponents)."""
        if not self.is_square():
            raise ValueError("Matrix must be square for exponentiation")
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")
        if exponent < 0:
            return self.inverse() ** (-exponent)
        
        result = Matrix.identity(self.rows)
        base = self
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = result * base
            base = base * base
            exponent //= 2
        
        return result
    
    # ==================== In-place Operations ====================
    
    def __iadd__(self, other):
        """In-place addition."""
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrix dimensions must match for addition")
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += other.data[i][j]
        elif isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += other
        return self
    
    def __isub__(self, other):
        """In-place subtraction."""
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrix dimensions must match for subtraction")
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] -= other.data[i][j]
        elif isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] -= other
        return self
    
    def __imul__(self, other):
        """In-place multiplication."""
        if isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= other
            return self
        else:
            raise TypeError("In-place multiplication only supports scalars")
    
    # ==================== Helper Methods for Operations ====================
    
    def _matrix_addition(self, other):
        """Add two matrices element-wise."""
        if self.shape != other.shape:
            raise ValueError("Matrix dimensions must match for addition")
        
        result = [[self.data[i][j] + other.data[i][j] 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def _scalar_addition(self, scalar):
        """Add a scalar to each element."""
        result = [[self.data[i][j] + scalar 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def _matrix_subtraction(self, other):
        """Subtract another matrix from this matrix."""
        if self.shape != other.shape:
            raise ValueError("Matrix dimensions must match for subtraction")
        
        result = [[self.data[i][j] - other.data[i][j] 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def _scalar_subtraction(self, scalar):
        """Subtract scalar from each element."""
        result = [[self.data[i][j] - scalar 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def _reverse_scalar_subtraction(self, scalar):
        """Subtract each element from scalar (scalar - matrix)."""
        result = [[scalar - self.data[i][j] 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def _scalar_multiplication(self, scalar):
        """Multiply each element by a scalar."""
        result = [[self.data[i][j] * scalar 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def _matrix_multiplication(self, other):
        """Multiply two matrices (dot product)."""
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply {self.shape} matrix by {other.shape} matrix")
        
        result = [[sum(self.data[i][k] * other.data[k][j] 
                      for k in range(self.cols))
                  for j in range(other.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def _scalar_division(self, scalar):
        """Divide each element by a scalar."""
        if scalar == 0:
            raise ValueError("Division by zero")
        result = [[self.data[i][j] / scalar 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    # ==================== Element-wise Operations ====================
    
    def hadamard_product(self, other):
        """Element-wise multiplication (Hadamard product)."""
        if self.shape != other.shape:
            raise ValueError("Matrix dimensions must match for Hadamard product")
        
        result = [[self.data[i][j] * other.data[i][j] 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    def element_wise_add(self, other):
        """Element-wise addition."""
        return self + other
    
    def element_wise_sub(self, other):
        """Element-wise subtraction."""
        return self - other
    
    def element_wise_div(self, other):
        """Element-wise division."""
        if self.shape != other.shape:
            raise ValueError("Matrix dimensions must match for element-wise division")
        
        result = [[self.data[i][j] / other.data[i][j] 
                  for j in range(self.cols)] 
                  for i in range(self.rows)]
        return Matrix(result)
    
    # ==================== Advanced Operations ====================
    
    def det(self):
        """Calculate the determinant of the matrix."""
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")
        
        n = self.rows
        if n == 1:
            return self.data[0][0]
        elif n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        elif n == 3:
            return (self.data[0][0] * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1])
                    - self.data[0][1] * (self.data[1][0] * self.data[2][2] - self.data[1][2] * self.data[2][0])
                    + self.data[0][2] * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0]))
        else:
            # Use LU decomposition for larger matrices
            return self._det_by_lu()
    
    def _det_by_lu(self):
        """Calculate determinant using LU decomposition."""
        n = self.rows
        # Create a copy to avoid modifying original
        mat = [row[:] for row in self.data]
        
        det = 1
        for i in range(n):
            # Find pivot
            pivot = i
            for j in range(i + 1, n):
                if abs(mat[j][i]) > abs(mat[pivot][i]):
                    pivot = j
            
            if abs(mat[pivot][i]) < 1e-10:
                return 0
            
            # Swap rows if needed
            if pivot != i:
                mat[i], mat[pivot] = mat[pivot], mat[i]
                det *= -1
            
            det *= mat[i][i]
            
            # Eliminate column
            for j in range(i + 1, n):
                factor = mat[j][i] / mat[i][i]
                for k in range(i, n):
                    mat[j][k] -= factor * mat[i][k]
        
        return det
    
    def inverse(self):
        """Calculate the inverse of the matrix."""
        if not self.is_square():
            raise ValueError("Inverse is only defined for square matrices")
        
        n = self.rows
        
        # Create augmented matrix [A|I]
        augmented = []
        for i in range(n):
            row = self.data[i] + Matrix.identity(n).data[i]
            augmented.append(row)
        
        # Gaussian elimination
        for i in range(n):
            # Find pivot
            pivot = i
            for j in range(i + 1, n):
                if abs(augmented[j][i]) > abs(augmented[pivot][i]):
                    pivot = j
            
            if abs(augmented[pivot][i]) < 1e-10:
                raise ValueError("Matrix is singular and cannot be inverted")
            
            # Swap rows
            augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
            
            # Scale pivot row
            scale = augmented[i][i]
            for j in range(2 * n):
                augmented[i][j] /= scale
            
            # Eliminate column
            for j in range(n):
                if j != i:
                    factor = augmented[j][i]
                    for k in range(2 * n):
                        augmented[j][k] -= factor * augmented[i][k]
        
        # Extract inverse
        inverse_data = [row[n:] for row in augmented]
        return Matrix(inverse_data)
    
    def cofactor(self):
        """Calculate the cofactor matrix."""
        if not self.is_square():
            raise ValueError("Cofactor is only defined for square matrices")
        
        n = self.rows
        if n == 1:
            return Matrix([[1]])
        
        result = []
        for i in range(n):
            row = []
            for j in range(n):
                # Get minor matrix
                minor = []
                for r in range(n):
                    if r != i:
                        minor_row = []
                        for c in range(n):
                            if c != j:
                                minor_row.append(self.data[r][c])
                        minor.append(minor_row)
                
                # Calculate minor determinant
                minor_det = Matrix(minor).det()
                
                # Apply cofactor sign
                cofactor = ((-1) ** (i + j)) * minor_det
                row.append(cofactor)
            result.append(row)
        
        return Matrix(result)
    
    def adjugate(self):
        """Calculate the adjugate (adjoint) matrix."""
        return self.cofactor().T()
    
    # ==================== Utility Methods ====================
    
    def is_square(self):
        """Check if the matrix is square."""
        return self.rows == self.cols
    
    def is_symmetric(self):
        """Check if the matrix is symmetric."""
        if not self.is_square():
            return False
        return self == self.T()
    
    def is_diagonal(self):
        """Check if the matrix is diagonal."""
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self.data[i][j] != 0:
                    return False
        return True
    
    def is_identity(self):
        """Check if the matrix is an identity matrix."""
        return self == Matrix.identity(self.rows)
    
    def is_zero(self):
        """Check if the matrix is a zero matrix."""
        return self == Matrix.zeros(self.rows, self.cols)
    
    def trace(self):
        """Calculate the trace of the matrix (sum of diagonal elements)."""
        if not self.is_square():
            raise ValueError("Trace is only defined for square matrices")
        
        return sum(self.data[i][i] for i in range(self.rows))
    
    def norm(self, order=2):
        """Calculate the norm of the matrix (Frobenius norm by default)."""
        if order == 2:
            # Frobenius norm
            return sum(self.data[i][j] ** 2 for i in range(self.rows) for j in range(self.cols)) ** 0.5
        elif order == 1:
            # Column sum norm
            return max(sum(abs(self.data[i][j]) for i in range(self.rows)) for j in range(self.cols))
        elif order == float('inf'):
            # Row sum norm
            return max(sum(abs(self.data[i][j]) for j in range(self.cols)) for i in range(self.rows))
        else:
            raise ValueError(f"Norm order {order} not supported")
    
    def rank(self):
        """Calculate the rank of the matrix using Gaussian elimination."""
        mat = [row[:] for row in self.data]
        rank = 0
        pivot_row = 0
        
        for col in range(self.cols):
            # Find pivot
            pivot = pivot_row
            while pivot < self.rows and abs(mat[pivot][col]) < 1e-10:
                pivot += 1
            
            if pivot < self.rows:
                mat[pivot], mat[pivot_row] = mat[pivot_row], mat[pivot]
                
                # Scale pivot row
                scale = mat[pivot_row][col]
                for j in range(col, self.cols):
                    mat[pivot_row][j] /= scale
                
                # Eliminate other rows
                for i in range(self.rows):
                    if i != pivot_row and abs(mat[i][col]) > 1e-10:
                        factor = mat[i][col]
                        for j in range(col, self.cols):
                            mat[i][j] -= factor * mat[pivot_row][j]
                
                pivot_row += 1
                rank += 1
        
        return rank
    
    def get_row(self, index):
        """Get a specific row as a list."""
        return self.data[index][:]
    
    def get_col(self, index):
        """Get a specific column as a list."""
        return [self.data[i][index] for i in range(self.rows)]
    
    def get_element(self, row, col):
        """Get a specific element."""
        return self.data[row][col]
    
    def set_element(self, row, col, value):
        """Set a specific element."""
        self.data[row][col] = value
    
    # ==================== String Representation ====================
    
    def __str__(self):
        """String representation of the matrix."""
        if self.rows == 0:
            return "Empty Matrix"
        
        # Calculate max width for alignment
        max_width = max(len(str(self.data[i][j])) for i in range(self.rows) for j in range(self.cols))
        
        lines = []
        for i, row in enumerate(self.data):
            if self.rows == 1:
                # Row vector
                row_str = "[" + " ".join(str(elem).rjust(max_width) for elem in row) + "]"
            elif i == 0:
                # First row
                row_str = "[" + " ".join(str(elem).rjust(max_width) for elem in row) + "]"
            elif i == self.rows - 1:
                # Last row
                row_str = " " + " ".join(str(elem).rjust(max_width) for elem in row) + "]"
            else:
                row_str = " " + " ".join(str(elem).rjust(max_width) for elem in row)
            lines.append(row_str)
        
        return "\n".join(lines)
    
    def __repr__(self):
        """Detailed representation of the matrix."""
        return f"Matrix({self.data}, shape={self.shape})"
    
    def __eq__(self, other):
        """Check equality of two matrices."""
        if not isinstance(other, Matrix):
            return False
        if self.shape != other.shape:
            return False
        return all(abs(self.data[i][j] - other.data[i][j]) < 1e-10 
                   for i in range(self.rows) for j in range(self.cols))
    
    def copy(self):
        """Create a deep copy of the matrix."""
        return Matrix([row[:] for row in self.data])
    
    # ==================== Matrix Functions (Standalone) ====================
    
    @staticmethod
    def concatenate(*matrices, axis=0):
        """Concatenate matrices along specified axis."""
        if len(matrices) < 2:
            raise ValueError("At least two matrices required for concatenation")
        
        if axis == 0:
            # Vertical concatenation (stack rows)
            if not all(matrices[0].cols == m.cols for m in matrices):
                raise ValueError("All matrices must have the same number of columns")
            result_data = []
            for m in matrices:
                result_data.extend(m.data)
            return Matrix(result_data)
        
        elif axis == 1:
            # Horizontal concatenation (stack columns)
            if not all(matrices[0].rows == m.rows for m in matrices):
                raise ValueError("All matrices must have the same number of rows")
            result_data = []
            for i in range(matrices[0].rows):
                row = []
                for m in matrices:
                    row.extend(m.data[i])
                result_data.append(row)
            return Matrix(result_data)
        
        else:
            raise ValueError("Axis must be 0 or 1")
    
    @staticmethod
    def vstack(*matrices):
        """Stack matrices vertically."""
        return Matrix.concatenate(*matrices, axis=0)
    
    @staticmethod
    def hstack(*matrices):
        """Stack matrices horizontally."""
        return Matrix.concatenate(*matrices, axis=1)


# ==================== Convenience Functions ====================

def zeros(rows, cols):
    """Create a matrix of zeros."""
    return Matrix.zeros(rows, cols)

def ones(rows, cols):
    """Create a matrix of ones."""
    return Matrix.ones(rows, cols)

def identity(n):
    """Create an identity matrix."""
    return Matrix.identity(n)

def random_matrix(rows, cols, min_val=0, max_val=1):
    """Create a matrix with random values."""
    return Matrix.random(rows, cols, min_val, max_val)

def diagonal(values):
    """Create a diagonal matrix."""
    return Matrix.diagonal(values)

def det(matrix):
    """Calculate determinant of a matrix."""
    return matrix.det()

def inverse(matrix):
    """Calculate inverse of a matrix."""
    return matrix.inverse()

def transpose(matrix):
    """Calculate transpose of a matrix."""
    return matrix.T()

def trace(matrix):
    """Calculate trace of a matrix."""
    return matrix.trace()

def rank(matrix):
    """Calculate rank of a matrix."""
    return matrix.rank()

def norm(matrix, order=2):
    """Calculate norm of a matrix."""
    return matrix.norm(order)


if __name__ == "__main__":
    # Demo code
    print("=" * 60)
    print("Matrix Library Demo")
    print("=" * 60)
    
    # Create matrices
    print("\n1. Creating Matrices:")
    print("-" * 40)
    
    A = Matrix([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])
    print("Matrix A:")
    print(A)
    
    B = Matrix([[9, 8, 7], 
                [6, 5, 4], 
                [3, 2, 1]])
    print("\nMatrix B:")
    print(B)
    
    # Basic operations
    print("\n2. Basic Operations:")
    print("-" * 40)
    
    print("\nA + B:")
    print(A + B)
    
    print("\nA - B:")
    print(A - B)
    
    print("\nA * 2 (scalar multiplication):")
    print(A * 2)
    
    print("\nA * B (matrix multiplication):")
    print(A * B)
    
    print("\nA * B (Hadamard/element-wise product):")
    print(A.hadamard_product(B))
    
    # Transpose
    print("\n3. Transpose:")
    print("-" * 40)
    print("A.T():")
    print(A.T())
    
    # Determinant
    print("\n4. Determinant:")
    print("-" * 40)
    C = Matrix([[1, 2], [3, 4]])
    print("Matrix C:")
    print(C)
    print(f"det(C) = {C.det()}")
    
    # Inverse
    print("\n5. Inverse:")
    print("-" * 40)
    D = Matrix([[4, 7], [2, 6]])
    print("Matrix D:")
    print(D)
    print("D.inverse():")
    print(D.inverse())
    
    # Identity and zeros
    print("\n6. Identity and Special Matrices:")
    print("-" * 40)
    print("Identity 3x3:")
    print(Matrix.identity(3))
    
    print("\nZeros 2x4:")
    print(Matrix.zeros(2, 4))
    
    print("\nOnes 3x2:")
    print(Matrix.ones(3, 2))
    
    # Properties
    print("\n7. Matrix Properties:")
    print("-" * 40)
    E = Matrix([[1, 2], [1, 2]])
    print("Matrix E:")
    print(E)
    print(f"Rank of E: {E.rank()}")
    print(f"Trace of E: {E.trace()}")
    print(f"Norm of E: {E.norm()}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
