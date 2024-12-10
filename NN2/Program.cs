﻿namespace NN2
{
    public class NeuralNetwork
    {
        public double[,] WeightsInputHidden;  // Input to Hidden Weights
        public double[,] WeightsHiddenOutput; // Hidden to Output Weights
        public double[] InputLayer;
        public double[] HiddenLayer;
        public double[] OutputLayer;
        public double[] Target;              // Target output
        public double LearningRate = 0.1;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            WeightsInputHidden = new double[inputSize, hiddenSize];
            WeightsHiddenOutput = new double[hiddenSize, outputSize];
            InputLayer = new double[inputSize];
            HiddenLayer = new double[hiddenSize];
            OutputLayer = new double[outputSize];
            Target = new double[outputSize];

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            Random rand = new Random();
            for (int i = 0; i < WeightsInputHidden.GetLength(0); i++)
            {
                for (int j = 0; j < WeightsInputHidden.GetLength(1); j++)
                {
                    WeightsInputHidden[i, j] = rand.NextDouble() * 0.2 - 0.1;
                }
            }

            for (int i = 0; i < WeightsHiddenOutput.GetLength(0); i++)
            {
                for (int j = 0; j < WeightsHiddenOutput.GetLength(1); j++)
                {
                    WeightsHiddenOutput[i, j] = rand.NextDouble() * 0.2 - 0.1;
                }
            }
        }

        public void Feedforward()
        {
            // Applying the Input to the Hidden Layer
            for (int j = 0; j < HiddenLayer.Length; j++)
            {
                HiddenLayer[j] = 0;
                for (int i = 0; i < InputLayer.Length; i++)
                {
                    HiddenLayer[j] += InputLayer[i] * WeightsInputHidden[i, j];
                }
                HiddenLayer[j] = Sigmoid(HiddenLayer[j]);
            }

            // Applying the Hidden to the Output Layer
            for (int j = 0; j < OutputLayer.Length; j++)
            {
                OutputLayer[j] = 0;
                for (int i = 0; i < HiddenLayer.Length; i++)
                {
                    OutputLayer[j] += HiddenLayer[i] * WeightsHiddenOutput[i, j];
                }
                OutputLayer[j] = Sigmoid(OutputLayer[j]);
            }
        }

        public void Backpropagation()
        {
            double[] outputDeltas = new double[OutputLayer.Length];
            for (int i = 0; i < OutputLayer.Length; i++)
            {
                double error = Target[i] - OutputLayer[i];
                outputDeltas[i] = error * SigmoidDerivative(OutputLayer[i]);
            }

            double[] hiddenDeltas = new double[HiddenLayer.Length];
            for (int i = 0; i < HiddenLayer.Length; i++)
            {
                double error = 0;
                for (int j = 0; j < OutputLayer.Length; j++)
                {
                    error += outputDeltas[j] * WeightsHiddenOutput[i, j];
                }
                hiddenDeltas[i] = error * SigmoidDerivative(HiddenLayer[i]);
            }

            // Update Weights for Hidden to Output
            for (int i = 0; i < HiddenLayer.Length; i++)
            {
                for (int j = 0; j < OutputLayer.Length; j++)
                {
                    WeightsHiddenOutput[i, j] += LearningRate * outputDeltas[j] * HiddenLayer[i];
                }
            }

            // Update Weights for Input to Hidden
            for (int i = 0; i < InputLayer.Length; i++)
            {
                for (int j = 0; j < HiddenLayer.Length; j++)
                {
                    WeightsInputHidden[i, j] += LearningRate * hiddenDeltas[j] * InputLayer[i];
                }
            }
        }

        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }
    }

}
