package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
	private List<InputNeuron> inputNeurons = new ArrayList<>();
	private List<WorkingNeuron> hiddenNeurons = new ArrayList<>();
	private List<WorkingNeuron> outputNeurons = new ArrayList<>();
	
	public WorkingNeuron createNewOutput() {
		WorkingNeuron wn = new WorkingNeuron();
		outputNeurons.add(wn);
		return wn;
	}
	
	public void createHiddenNeurons(int amount) {
		for(int i = 0; i<amount; i++)
			hiddenNeurons.add(new WorkingNeuron());
	}
	
	public InputNeuron createNewInput() {
		InputNeuron in = new InputNeuron();
		inputNeurons.add(in);
		return in;
	}
	
	public void reset() {
		for(WorkingNeuron wn : hiddenNeurons)
			wn.reset();
		for(WorkingNeuron wn : outputNeurons)
			wn.reset();
	}
	
	public void backpropagation(float[] shoulds, float epsilon) {
		if(shoulds.length != outputNeurons.size())
			throw new IllegalArgumentException();
		
		reset();
		
		for(int i=0; i<shoulds.length; i++)
			outputNeurons.get(i).calcSmallDelta(shoulds[i]);
		
		if(hiddenNeurons.size() > 0)
			for(int i=0; i<shoulds.length; i++)
				outputNeurons.get(i).backpropagateSmallDelta();
		
		for(int i=0; i<shoulds.length; i++) 
			outputNeurons.get(i).deltaLearning(epsilon);
		
		for(int i=0; i<shoulds.length; i++) 
			hiddenNeurons.get(i).deltaLearning(epsilon);
	}
	
	public void createFullMesh() {
		if(hiddenNeurons.size() == 0) {
			for(WorkingNeuron wn : outputNeurons)
				for(InputNeuron in : inputNeurons)
					wn.addConnection(new Connection(in, 0));
			
		} else {
			for (WorkingNeuron wn : outputNeurons)
				for (WorkingNeuron hidden : hiddenNeurons)
					wn.addConnection(new Connection(hidden, 0));
			
			for (WorkingNeuron hidden : hiddenNeurons)
				for (InputNeuron in : inputNeurons)
					hidden.addConnection(new Connection(in, 0));
		}
	}
	
	public void createFullMesh(float... weights) {
		if (hiddenNeurons.size() == 0) {
			if (weights.length != inputNeurons.size() * outputNeurons.size())
				throw new RuntimeException();

			int index = 0;

			for (WorkingNeuron wn : outputNeurons)
				for (InputNeuron in : inputNeurons)
					wn.addConnection(new Connection(in, weights[index++]));
		} else {
			if (weights.length != (inputNeurons.size() + outputNeurons.size()) * hiddenNeurons.size())
				throw new RuntimeException();
			
			int index = 0;
			
			for (WorkingNeuron hidden : hiddenNeurons)
				for (InputNeuron in : inputNeurons)
					hidden.addConnection(new Connection(in, weights[index++]));
			
			for (WorkingNeuron out : outputNeurons)
				for (WorkingNeuron hidden : hiddenNeurons)
					out.addConnection(new Connection(hidden, weights[index++]));
		}
	}
}
