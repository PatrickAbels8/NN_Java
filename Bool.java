package NeuralNetwork.ActivationFunctions;

public class Bool implements ActivationFunction {

	@Override
	public float activation(float input) {
		return input<0 ? 0: 1;
	}

	@Override
	public float derivative(float input) {
		return 1;
	}

}
