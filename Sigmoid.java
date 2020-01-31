package NeuralNetwork.ActivationFunctions;

public class Sigmoid implements ActivationFunction {

	@Override
	public float activation(float input) {
		return (float)(1f / (1f + Math.pow(Math.E, -input)));
	}

	@Override
	public float derivative(float input) {
		return activation(input)*(1-activation(input));
	}

}
