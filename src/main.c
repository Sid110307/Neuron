#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 20
#define HIDDEN_SIZE 40
#define OUTPUT_SIZE 20
#define LEARNING_RATE 0.1
#define EPOCHS 10000

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x)
{
	double sigX = sigmoid(x);
	return sigX * (1.0 - sigX);
}

void initWeights(double* weights, int size)
{
	for (int i = 0; i < size; i++) weights[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
}

void forward(const double* input, double* hidden, double* output, const double* inputWeights,
			 const double* hiddenWeights)
{
	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < INPUT_SIZE; j++) sum += input[j] * inputWeights[j * HIDDEN_SIZE + i];
		hidden[i] = sigmoid(sum);
	}

	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < HIDDEN_SIZE; j++) sum += hidden[j] * hiddenWeights[j * OUTPUT_SIZE + i];
		output[i] = sigmoid(sum);
	}
}

void backPropagation(const double* input, double* hidden, double* output, const double* target, double* inputWeights,
					 double* hiddenWeights)
{
	double outputError[OUTPUT_SIZE];
	double hiddenError[HIDDEN_SIZE];

	for (int i = 0; i < OUTPUT_SIZE; i++) outputError[i] = (target[i] - output[i]) * sigmoidDerivative(output[i]);
	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		double errorSum = 0.0;
		for (int j = 0; j < OUTPUT_SIZE; j++) errorSum += outputError[j] * hiddenWeights[i * OUTPUT_SIZE + j];
		hiddenError[i] = errorSum * sigmoidDerivative(hidden[i]);
	}

	for (int i = 0; i < HIDDEN_SIZE; i++)
		for (int j = 0; j < OUTPUT_SIZE; j++)
			hiddenWeights[i * OUTPUT_SIZE + j] += LEARNING_RATE * outputError[j] * hidden[i];

	for (int i = 0; i < INPUT_SIZE; i++)
		for (int j = 0; j < HIDDEN_SIZE; j++)
			inputWeights[i * HIDDEN_SIZE + j] += LEARNING_RATE * hiddenError[j] * input[i];
}

void train(double* input, double* target, double* inputWeights, double* hiddenWeights)
{
	for (int epoch = 0; epoch < EPOCHS; epoch++)
	{
		double hidden[HIDDEN_SIZE];
		double output[OUTPUT_SIZE];

		forward(input, hidden, output, inputWeights, hiddenWeights);
		backPropagation(input, hidden, output, target, inputWeights, hiddenWeights);
	}
}

int main()
{
	srand(time(NULL));
	double input[INPUT_SIZE] = {
			1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
			1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0
	};

	double target[OUTPUT_SIZE] = {
			0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
			0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
	};

	double inputWeights[INPUT_SIZE * HIDDEN_SIZE];
	double hiddenWeights[HIDDEN_SIZE * OUTPUT_SIZE];

	initWeights(inputWeights, INPUT_SIZE * HIDDEN_SIZE);
	initWeights(hiddenWeights, HIDDEN_SIZE * OUTPUT_SIZE);

	train(input, target, inputWeights, hiddenWeights);

	double hidden[HIDDEN_SIZE];
	double output[OUTPUT_SIZE];

	forward(input, hidden, output, inputWeights, hiddenWeights);
	for (int i = 0; i < INPUT_SIZE; i++)
		printf("%i: %f -> %f (%.2f%% accuracy)\n", i, input[i], output[i], output[i] * 100.0);

	return EXIT_SUCCESS;
}
