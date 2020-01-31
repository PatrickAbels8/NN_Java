package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import NeuralNetwork.ActivationFunctions.ActivationFunction;

public class WorkingNeuron extends Neuron{
	private List<Connection> connections = new ArrayList<>();
	private ActivationFunction activationFunction = ActivationFunction.ActivationSigmoid;
	private float smallDelta = 0;
	private float value = 0;
	private boolean valueClean = false;
	
	@Override
	public float getValue() {
		if (!valueClean) {
			float sum = 0;
			for (Connection c : connections)
				sum += c.getValue();

			value = activationFunction.activation(sum);
			valueClean = true;
		}
		
		return value;
		
	}
	
	public void reset() {
		smallDelta = 0;
		valueClean = false;
	}
	
	public void addConnection(Connection c) {
		connections.add(c);
	}
	
	public void backpropagateSmallDelta() {
		for(Connection c : connections) {
			if(c.getNeuron() instanceof WorkingNeuron) {
				WorkingNeuron wn = (WorkingNeuron)c.getNeuron();
				wn.smallDelta += this.smallDelta * c.getWeight();
			}
			
		}
	}
	
	public void calcSmallDelta(float should) {
		smallDelta = should - getValue();
	}
	
	public void deltaLearning(float epsilon) {
		float factor = activationFunction.derivative(getValue()) * epsilon * smallDelta;
		for(int i=0; i<connections.size(); i++) {
			float bigDelta = factor * connections.get(i).getNeuron().getValue();
			connections.get(i).addWeight(bigDelta);
		}
	}
	
	
}
