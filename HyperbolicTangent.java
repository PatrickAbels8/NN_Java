package NeuralNetwork.ActivationFunctions;

public class HyperbolicTangent implements ActivationFunction {

	@Override
	public float activation(float input) {
		double epx = Math.pow(Math.E, input);
		double enx = Math.pow(Math.E, -input);
		
		return (float)((epx - enx)/(epx + enx));
	}

	@Override
	public float derivative(float input) {
		return 1-activation(input)*activation(input);
	}

}
