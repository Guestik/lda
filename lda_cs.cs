using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text.RegularExpressions;
using Accord.Math;

namespace lda_cs
{
    class Program
    {
        static void Main(string[] args)
        {
            //Training data loading
            List<string> classes = new List<string>(); //List of classes
            List<double[]> dataX = new List<double[]>(); //All data of all classes together
            List<double[]> dataX_1 = new List<double[]>(); //1st class
            List<double[]> dataX_2 = new List<double[]>(); //2nd class
            List<double[]> dataX_3 = new List<double[]>(); //3rd class
            List<double[]> dataX_4 = new List<double[]>(); //4th class
            List<double[]> dataX_5 = new List<double[]>(); //5th class
            List<double[]> dataX_6 = new List<double[]>(); //6th class
            string line;
            Regex dotPattern = new Regex("[.]"); //Need to replace ',' because otherwise can not parse numbers
            StreamReader sr = new StreamReader("training_data_classification.txt"); //Input file
            line = sr.ReadLine(); //Second line of the input
            while (line != null) //While there is something to read from the file
            {
                string[] split = line.Split(',');

                for (int tempik = 0; tempik < split.Length; tempik++)
                    split[tempik] = dotPattern.Replace(split[tempik], ",");

                double[] itemik = new double[split.Length - 1]; //Minus 1 because the last one is a class
                for (int i = 0; i < split.Count() - 1; i++)
                    itemik[i] = double.Parse(split[i]);

                classes.Add(split[split.Count() - 1]);

                dataX.Add(itemik);
                if (split[split.Count() - 1].Trim() == "1")
                    dataX_1.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "2")
                    dataX_2.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "3")
                    dataX_3.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "4")
                    dataX_4.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "5")
                    dataX_5.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "6")
                    dataX_6.Add(itemik);
                line = sr.ReadLine(); //Read next line
            }
            sr.Close(); //Close the file

            int numOfDimensions = dataX_1[0].Length; //Number of input dimensions
            int newDimensions = 2; //Number of dimensions to be reduced to
            int numberOfData = dataX.Count(); //Number of all data

            //Calculate mean of all data
            double[] mi = Mi(dataX, numOfDimensions);

            //Calculate mean of each class
            double[] mi_1 = Mi(dataX_1, numOfDimensions);
            double[] mi_2 = Mi(dataX_2, numOfDimensions);
            double[] mi_3 = Mi(dataX_3, numOfDimensions);
            double[] mi_4 = Mi(dataX_4, numOfDimensions);
            double[] mi_5 = Mi(dataX_5, numOfDimensions);
            double[] mi_6 = Mi(dataX_6, numOfDimensions);

            List<double[]> listOfMeans = new List<double[]>();
            listOfMeans.Add(mi_1);
            listOfMeans.Add(mi_2);
            listOfMeans.Add(mi_3);
            listOfMeans.Add(mi_4);
            listOfMeans.Add(mi_5);
            listOfMeans.Add(mi_6);

            //Calculate variance of each class
            double[] variance_1 = Variance(dataX_1, numOfDimensions, mi_1);
            double[] variance_2 = Variance(dataX_2, numOfDimensions, mi_2);
            double[] variance_3 = Variance(dataX_3, numOfDimensions, mi_3);
            double[] variance_4 = Variance(dataX_4, numOfDimensions, mi_4);
            double[] variance_5 = Variance(dataX_5, numOfDimensions, mi_5);
            double[] variance_6 = Variance(dataX_6, numOfDimensions, mi_6);

            //Calculate covariance of each class
            double[,] coverianceMatrix_1 = Covariance(dataX_1, numOfDimensions, mi_1, variance_1);
            double[,] coverianceMatrix_2 = Covariance(dataX_2, numOfDimensions, mi_2, variance_2);
            double[,] coverianceMatrix_3 = Covariance(dataX_3, numOfDimensions, mi_3, variance_3);
            double[,] coverianceMatrix_4 = Covariance(dataX_4, numOfDimensions, mi_4, variance_4);
            double[,] coverianceMatrix_5 = Covariance(dataX_5, numOfDimensions, mi_5, variance_5);
            double[,] coverianceMatrix_6 = Covariance(dataX_6, numOfDimensions, mi_6, variance_6);

            //Calculate Sw
            double[,] Sw = Elementwise.Add(Elementwise.Add(Elementwise.Add(coverianceMatrix_1, coverianceMatrix_2), Elementwise.Add(coverianceMatrix_3, coverianceMatrix_4)), Elementwise.Add(coverianceMatrix_5, coverianceMatrix_6));

            //Calculate M
            double[,] M = Matrix.Transpose(Elementwise.Divide(Elementwise.Add(Elementwise.Add(Elementwise.Add(mi_1, mi_2), Elementwise.Add(mi_3, mi_4)), Elementwise.Add(mi_5, mi_6)), 6));

            //Calculate Sb
            double[,] Sb = new double[8, 8];
            for (int i = 0; i < 6; i++)
            {
                double[,] miMinusM = Elementwise.Subtract(Matrix.Transpose(listOfMeans[i]), M);
                Sb = Elementwise.Add(Sb, Matrix.Dot(miMinusM, Matrix.Transpose(miMinusM)));
            }

            //Get sorted eigenvelues and eigenvectors
            Accord.Math.Decompositions.EigenvalueDecomposition dve = new Accord.Math.Decompositions.EigenvalueDecomposition(Matrix.Dot(Matrix.Inverse(Sw), Sb), false, true);

            //Calculate projection matrix using 2 eigenvectors associated with 2 largest eigenvalues
            double[,] W = new double[dve.Eigenvectors.GetLength(1), newDimensions];
            for (int i = 0; i < dve.Eigenvectors.GetLength(1); i++)
            {
                for (int q = 0; q < newDimensions; q++)
                {
                    W[i, q] = dve.Eigenvectors[i, q];
                }
            }

            //Calculate 'z'
            List<double[]> z = new List<double[]>();
            for (int i = 0; i < numberOfData; i++)
                z.Add(Matrix.Dot(Matrix.Transpose(W), Elementwise.Subtract(dataX[i], mi)));

            //Export reduced dataset with corresponding classes to the file
            FileStream stream = new FileStream("answer_data_lda.txt", FileMode.Create);
            StreamWriter file = new StreamWriter(stream);
            using (file)
            {
                for (int i = 0; i < z.Count(); i++)
                    file.WriteLine("{0};{1};{2}", z[i][0], z[i][1], classes[i]);
            }
        }

        public static double[] Mi(List<double[]> dataX, int pocetDimenzi)
        {
            double[] mi = new double[pocetDimenzi];
            for (int y = 0; y < dataX.Count; y++) //For every N
                for (int x = 0; x < dataX[y].Length; x++) //For every element in transaction
                    mi[x] += dataX[y][x];

            for (int d = 0; d < pocetDimenzi; d++)
            {
                mi[d] = mi[d] / dataX.Count;
            }
            return mi;
        }

        public static double[] Variance(List<double[]> dataX, int dimensions, double[] mi)
        {
            double[] varOfEachColumn = new double[dimensions];
            for (int i = 0; i < dimensions; i++)
            {
                for (int j = 0; j < dataX.Count; j++)
                {
                    varOfEachColumn[i] += Math.Pow(dataX[j][i] - mi[i], 2);
                }
            }
            for (int d = 0; d < varOfEachColumn.Length; d++)
            {
                varOfEachColumn[d] = varOfEachColumn[d] / dataX.Count;
            }
            return varOfEachColumn;
        }

        public static double Coveriance(double[] dimenzeA, double meanA, double[] dimenzeB, double meanB, int numberOfData)
        {
            double result = 0;
            for (int a = 0; a < dimenzeA.Length; a++) //Both dimensions are same lenght, so it doesnt matter which I put here
                result += (dimenzeA[a] - meanA) * (dimenzeB[a] - meanB);
            return result / numberOfData;
        }

        public static double[,] Covariance(List<double[]> dataX, int pocetDimenzi, double[] mi, double[] varKazdehoSloupce)
        {
            //Covariance (d*d matrix)
            double[,] coverianceMatrix = new double[pocetDimenzi, pocetDimenzi];
            double[] dimenzeA_temp = new double[dataX.Count];
            double[] dimenzeB_temp = new double[dataX.Count];

            for (int x = 0; x < pocetDimenzi; x++)
            {
                for (int y = 0; y < pocetDimenzi; y++)
                {
                    if (x == y) //Variance on the diagonal
                    {
                        coverianceMatrix[x, y] = varKazdehoSloupce[x];
                    }
                    else
                    {
                        for (int i = 0; i < dataX.Count; i++)
                        {
                            dimenzeA_temp[i] = dataX[i][x];
                            dimenzeB_temp[i] = dataX[i][y];
                        }

                        coverianceMatrix[x, y] = Coveriance(dimenzeA_temp, mi[x], dimenzeB_temp, mi[y], dataX.Count);
                    }
                }
            }
            return coverianceMatrix;
        }
    }
}
