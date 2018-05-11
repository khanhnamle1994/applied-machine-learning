# 5 Reasons to Learn Linear Algebra for Machine Learning
Linear algebra is a field of mathematics that could be called the mathematics of data.

It is undeniably a pillar of the field of machine learning, and many recommend it as a prerequisite subject to study prior to getting started in machine learning. This is misleading advice, as linear algebra makes more sense to a practitioner once they have a context of the applied machine learning process in which to interpret it.

## 1. You Need to Learn Linear Algebra Notation
You need to be able to read and write vector and matrix notation.

Algorithms are described in books, papers and on websites using vector and matrix notation.

Linear algebra is the mathematics of data and the notation allows you to describe operations on data precisely with specific operators.

You need to be able to read and write this notation. This skill will allow you to:

* Read descriptions of existing algorithms in textbooks.
* Interpret and implement descriptions of new methods in research papers.
* Concisely describe your own methods to other practitioners.

Further, programming languages such as Python offer efficient ways of implementing linear algebra notation directly.

An understanding of the notation and how it is realized in your language or library will allow for shorter and perhaps more efficient implementations of machine learning algorithms.

## 2. You Need to Learn Linear Algebra Arithmetic
In partnership with the notation of linear algebra are the arithmetic operations performed.

You need to know how to add, subtract, and multiply scalars, vectors, and matrices.

A challenge for newcomers to the field of linear algebra are operations such as matrix multiplication and tensor multiplication that are not implemented as the direct multiplication of the elements of these structures, and at first glance appear nonintuitive.

Again, most if not all of these operations are implemented efficiently and provided via API calls in modern linear algebra libraries.

An understanding of how vector and matrix operations are implemented is required as a part of being able to effectively read and write matrix notation.

## 3. You Need to Learn Linear Algebra for Statistics
You must learn linear algebra in order to be able to learn statistics. Especially multivariate statistics.

Statistics and data analysis are another pillar field of mathematics to support machine learning. They are primarily concerned with describing and understanding data. As the mathematics of data, linear algebra has left its fingerprint on many related fields of mathematics, including statistics.

In order to be able to read and interpret statistics, you must learn the notation and operations of linear algebra.

Modern statistics uses both the notation and tools of linear algebra to describe the tools and techniques of statistical methods. From vectors for the means and variances of data, to covariance matrices that describe the relationships between multiple Gaussian variables.

The results of some collaborations between the two fields are also staple machine learning methods, such as the Principal Component Analysis, or PCA for short, used for data reduction.

## 4. You Need to Learn Matrix Factorization
Building on notation and arithmetic is the idea of matrix factorization, also called matrix decomposition.

You need to know how to factorize a matrix and what it means.

Matrix factorization is a key tool in linear algebra and used widely as an element of many more complex operations in both linear algebra (such as the matrix inverse) and machine learning (least squares).

Further, there are a range of different matrix factorization methods, each with different strengths and capabilities, some of which you may recognize as “machine learning” methods, such as Singular-Value Decomposition, or SVD for short, for data reduction.

In order to read and interpret higher-order matrix operations, you must understand matrix factorization.

## 5. You Need to Learn Linear Least Squares
You need to know how to use matrix factorization to solve linear least squares.

Linear algebra was originally developed to solve systems of linear equations. These are cases where there are more equations than there are unknown variables (e.g. coefficients). As a result, they are challenging to solve arithmetically because there is no single solution as there is no line or plane can fit the data without some error.

Problems of this type can be framed as the minimization of squared error, called least squares, and can be recast in the language of linear algebra, called linear least squares.

Linear least squares problems can be solved efficiently on computers using matrix operations such as matrix factorization.

Least squares is most known for its role in the solution to linear regression models, but also plays a wider role in a range of machine learning algorithms.

In order to understand and interpret these algorithms, you must understand how to use matrix factorization methods to solve least squares problems.

# Where to Start in Linear Algebra?
Perhaps now you are motivated to take a step into the field of linear algebra.

I would caution you to not take a straight course on linear algebra. It is a big field, and not all of it will be relevant or applicable to you as a machine learning practitioner, at least not in the beginning.

I would recommend a staggered approach and starting with the following areas of linear algebra that are relevant to machine learning.

## Vector and Matrix Notation

* [Linear Algebra Cheat Sheet for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Linear_Algebra_Cheat_Sheet.ipynb)

* [10 Examples of Linear Algebra in Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/10_Examples_of_Linear_Algebra_in_ML.ipynb)

* [Basics of Mathematical Notation for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Basic_of_Mathematical_Notation_for_ML.ipynb)

* [Introduction to Matrix Types in Linear Algebra for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Introduction_to_Matrix_Types_in_Linear_Algebra_for_ML.ipynb)

## Vector and Matrix Arithmetic

* [A Gentle Introduction to Vectors for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Gentle_Introduction_to_Vectors_for_ML.ipynb)

* [Introduction to Matrices and Matrix Arithmetic for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Introduction_to_Matrices_and_Matrix_Arithmetic_for_ML.ipynb)

* [How to Index, Slice and Reshape NumPy Arrays for Machine Learning in Python](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Index_Slice_Reshape_NumPy_Arrays_for_ML_in_Python.ipynb)

* [A Gentle Introduction to Broadcasting with NumPy Arrays](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Broadcasting_with_Numpy_Arrays.ipynb)

* [A Gentle Introduction to N-Dimensional Arrays in Python with NumPy](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/A_Gentle_Introduction_to_N-Dimensional_Arrays_in_Python_with_NumPy.ipynb)

* [A Gentle Introduction to Tensors for Machine Learning with NumPy](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/A_Gentle_Introduction_to_Tensors_for_ML_with_Numpy.ipynb)

* [A Gentle Introduction to Sparse Matrices for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Sparse_Matrices_for_Machine_Learning.ipynb)

## Multivariate Statistics

* [A Gentle Introduction to Expected Value, Variance, and Covariance with NumPy](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/A_Gentle_Introduction_to_Expected_Value_Variance_Covariance_with_NumPy.ipynb)

## Matrix Factorization

* [A Gentle Introduction to Matrix Operations for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/A_Gentle_Introduction_to_Matrix_Operations_for_ML.ipynb)

* [A Gentle Introduction to Matrix Factorization for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Introduction_to_Matrix_Factorization.ipynb)

* [Gentle Introduction to Eigendecomposition, Eigenvalues, and Eigenvectors for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Introduction_to_Eigendecomposition_Eigenvalues_Eigenvectors.ipynb)

* [A Gentle Introduction to Singular-Value Decomposition for Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/A_Gentle_Introduction_to_SVD_for_ML.ipynb)

* [How to Calculate the Principal Component Analysis from Scratch in Python](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Calculate_PCA_from_Scratch_in_Python.ipynb)

## Linear Least Squares

* [Gentle Introduction to Vector Norms in Machine Learning](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Gentle_Introduction_to_Vector_Norms_in_ML.ipynb)

* [How to Solve Linear Regression Using Linear Algebra](https://github.com/khanhnamle1994/applied-machine-learning/blob/master/Linear-Algebra/Solve_Linear_Regression_using_Linear_Algebra.ipynb)

I would consider this the minimum linear algebra required to be an effective machine learning practitioner.

You can go deeper and learn how the operations were derived, which in turn may deepen your understanding and effectiveness in some aspects of applied machine learning, but it may be beyond the point of diminishing returns for most practitioners, at least in terms of the day-to-day activities of the average machine learning practitioner.
