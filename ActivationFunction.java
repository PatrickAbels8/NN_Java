package NeuralNetwork.ActivationFunctions;

public interface ActivationFunction {
	public static Bool ActivationBool = new Bool();
	public static Identity ActivationIdentity = new Identity();
	public static Sigmoid ActivationSigmoid = new Sigmoid();
	public static HyperbolicTangent ActivationHyperbolicTangent = new HyperbolicTangent();
	
	public float activation(float input);
	public float derivative(float input);
}
