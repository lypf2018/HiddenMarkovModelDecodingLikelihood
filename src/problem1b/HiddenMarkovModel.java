/**
 * 
 */
package problem1b;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * HiddenMarkovModel including likelihood calculation and decoding method
 * 
 * @author yzc
 *
 * @param <T> the type of state in the model
 * @param <S> the type of observation in the model
 */
public class HiddenMarkovModel<T, S> {

	private T startState;
	private T endState;

	private List<T> stateList;
	private Map<T, Map<T, Double>> transitionProbabilityMap; // A matrix
	private Map<T, Map<S, Double>> observationLikelihoodMap; // B matrix

	/**
	 * @return the startState
	 */
	public T getStartState() {
		return startState;
	}

	/**
	 * @param startState the startState to set
	 */
	public void setStartState(T startState) {
		this.startState = startState;
	}

	/**
	 * @return the endState
	 */
	public T getEndState() {
		return endState;
	}

	/**
	 * @param endState the endState to set
	 */
	public void setEndState(T endState) {
		this.endState = endState;
	}

	/**
	 * @return the stateList
	 */
	public List<T> getStateList() {
		return stateList;
	}

	/**
	 * @return the transitionProbabilityMap
	 */
	public Map<T, Map<T, Double>> getTransitionProbabilityMap() {
		return transitionProbabilityMap;
	}

	/**
	 * @return the observationLikelihoodMap
	 */
	public Map<T, Map<S, Double>> getObservationLikelihoodMap() {
		return observationLikelihoodMap;
	}

	/**
	 * After using this constructor, you should set start and end state before setting up the model
	 */
	public HiddenMarkovModel() {
		stateList = new ArrayList<>();
		transitionProbabilityMap = new HashMap<>();
		observationLikelihoodMap = new HashMap<>();
	}

	/**
	 * After using this constructor, you should set up the model before using it
	 * @param startState Start state
	 * @param endState End state
	 */
	public HiddenMarkovModel(T startState, T endState) {
		this.startState = startState;
		this.endState = endState;
		stateList = new ArrayList<>();
		transitionProbabilityMap = new HashMap<>();
		observationLikelihoodMap = new HashMap<>();
	}

	/**
	 * @param startState Start state
	 * @param endState End state
	 * @param stateList Normal state list 
	 * @param transitionProbabilityMap Transition Probability matrix, A matrix
	 * @param observationLikelihoodMap Observation Likelihood matrix, B matrix
	 */
	public HiddenMarkovModel(T startState, T endState, List<T> stateList, Map<T, Map<T, Double>> transitionProbabilityMap, Map<T, Map<S, Double>> observationLikelihoodMap) {
		this.startState = startState;
		this.endState = endState;
		this.stateList = stateList;
		this.transitionProbabilityMap = transitionProbabilityMap;
		this.observationLikelihoodMap = observationLikelihoodMap;
	}

	/**
	 * Hidden Markov Model Likelihood process. Input observation sequence, and output the likelihood probability
	 * @param observationSequence Observation sequence
	 * @return the likelihood probability
	 */
	public double likelihood(List<S> observationSequence) {
		// Forward algorithm
		// Boundary condition (Initialization)
		List<Double> probabilityInStateList = new ArrayList<>();
		for (int i = 0; i < stateList.size(); i++) {
			double startProbability = 1.0;
			double transitionProbability = transitionProbabilityMap.get(startState).get(stateList.get(i));
			double observationLikelihood = observationLikelihoodMap.get(stateList.get(i)).get(observationSequence.get(0));
			double currentProbability = startProbability * transitionProbability * observationLikelihood;
			probabilityInStateList.add(currentProbability);
		}

		// Iteration (Recursion)
		for (int i = 1; i < observationSequence.size(); i++) {
			List<Double> previousProbabilityInStateList = probabilityInStateList;
			probabilityInStateList = new ArrayList<>();
			for (int j = 0; j < stateList.size(); j++) {
				T currentState = stateList.get(j);
				double sumProbability = 0.0;
				for (int k = 0; k < previousProbabilityInStateList.size(); k++) {
					T previousState =  stateList.get(k);
					double previousProbability = previousProbabilityInStateList.get(k);
					double transitionProbability = transitionProbabilityMap.get(previousState).get(currentState);
					double observationLikelihood = observationLikelihoodMap.get(currentState).get(observationSequence.get(i));
					double currentProbability = previousProbability * transitionProbability * observationLikelihood;
					sumProbability += currentProbability;
				}
				probabilityInStateList.add(sumProbability);
			}
		}

		// Termination
		double likelihood = 0.0;
		for (int i = 0; i < probabilityInStateList.size(); i++) {
			double probability = probabilityInStateList.get(i) * transitionProbabilityMap.get(stateList.get(i)).getOrDefault(endState, 1.0);
			likelihood += probability;
		}
		return likelihood;
	}

	/**
	 * Hidden Markov Model Decoding process. Input observation sequence, and output most likely state sequence
	 * @param observationSequence Observation sequence
	 * @return Most likely state sequence
	 */
	public List<T> decoding(List<S> observationSequence) {
		List<List<Integer>> backtraceList = new ArrayList<>();
		List<Double> probabilityInStateList = viterbi(observationSequence, backtraceList);

		// Get last state index
		double maxProbability = 0.0;
		int lastStateIndex = 0;
		for (int i = 0; i < probabilityInStateList.size(); i++) {
			double probability = probabilityInStateList.get(i) * transitionProbabilityMap.get(stateList.get(i)).getOrDefault(endState, 1.0);
			if (probability >= maxProbability) {
				maxProbability = probability;
				lastStateIndex = i;
			}
		}

		// Back trace state sequence
		List<T> mostLikelyStateSequence = new ArrayList<>();
		mostLikelyStateSequence.add(stateList.get(lastStateIndex));
		int stateIndex = lastStateIndex;
		for (int i = backtraceList.size() - 1; i >= 0; i--) {
			stateIndex = backtraceList.get(i).get(stateIndex);
			mostLikelyStateSequence.add(stateList.get(stateIndex));
		}
		Collections.reverse(mostLikelyStateSequence);
		return mostLikelyStateSequence;
	}

	/**
	 * Dynamic programming for viterbi algorithm
	 * @param observationSequence Observation sequence
	 * @param backtraceList List of each back pointer for each state of Observation Index
	 * @return Probability for each state of last observation
	 */
	private List<Double> viterbi(List<S> observationSequence, List<List<Integer>> backtraceList) {
		// Boundary condition (Initialization)
		List<Double> probabilityInStateList = new ArrayList<>();
		for (int i = 0; i < stateList.size(); i++) {
			double startProbability = 1.0;
			double transitionProbability = transitionProbabilityMap.get(startState).get(stateList.get(i));
			double observationLikelihood = observationLikelihoodMap.get(stateList.get(i)).get(observationSequence.get(0));
			double currentProbability = startProbability * transitionProbability * observationLikelihood;
			probabilityInStateList.add(currentProbability);
		}

		// Iteration (Recursion)
		for (int i = 1; i < observationSequence.size(); i++) {
			List<Double> previousProbabilityInStateList = probabilityInStateList;
			probabilityInStateList = new ArrayList<>();
			List<Integer> currentObservationBacktraceList = new ArrayList<>();
			for (int j = 0; j < stateList.size(); j++) {
				T currentState = stateList.get(j);
				double maxProbability = 0.0;
				int stateBackPointer = 0;
				for (int k = 0; k < previousProbabilityInStateList.size(); k++) {
					T previousState =  stateList.get(k);
					double previousProbability = previousProbabilityInStateList.get(k);
					double transitionProbability = transitionProbabilityMap.get(previousState).get(currentState);
					double observationLikelihood = observationLikelihoodMap.get(currentState).get(observationSequence.get(i));
					double currentProbability = previousProbability * transitionProbability * observationLikelihood;
					if (currentProbability >= maxProbability) {
						maxProbability = currentProbability;
						stateBackPointer = k;
					}
				}
				probabilityInStateList.add(maxProbability);
				currentObservationBacktraceList.add(stateBackPointer);
			}
			backtraceList.add(currentObservationBacktraceList);
		}
		return probabilityInStateList;
	}
}
