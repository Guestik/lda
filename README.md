# lda
LDA in C# (ML model)

Linear discriminant analysis (LDA) process details:

1. Importing data
2. Calculating mean of all data
3. Calculating mean of each class
4. Calculating variance of each class
5. Calculating covariance of each class
6. Calculating Sw and Sb
7. Getting sorted eigenvalues and eigenvectors from Sw^(-1)*Sb
8. Calculating ‘W’ transformation matrix using 2 eigenvectors associated with 2 largest eigenvalues
9. Calculating ‘z’
10. Exporting reduced dataset with corresponding classes
11. Plot the data (using the same .py code as for the PCA)

Plot example:

![image](https://github.com/Guestik/lda/assets/18994179/bc197c03-770f-4517-8da5-18ef2b75a784)
