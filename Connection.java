package NeuralNetwork;

public class Connection {
	private Neuron neuron;
	private float weight;
	private float momentum = 0;
	
	public Connection(Neuron neuron, float weight) {
		this.neuron = neuron;
		this.weight = weight;
	}
	
	public float getValue() {
		return neuron.getValue()*weight;
	}
	
	public float getWeight() {
		return weight;
	}
	
	public void addWeight(float weightDelta) {
		momentum += weightDelta;
		momentum *= 0.9f;
		weight += weightDelta + momentum;
	}
	
	public Neuron getNeuron() {
		return neuron;
	}
}
