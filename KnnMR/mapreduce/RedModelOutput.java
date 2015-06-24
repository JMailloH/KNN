package org.apache.mahout.classifier.KnnMR.mapreduce;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;


// Print a PrototypeSet! .

/**
 * Print a reduced set as a PrototypeSet.
 */
public class RedModelOutput implements Writable, Cloneable {

  private ArrayList<int []> predictedRightClass;

 // private int[] predictions;

  public RedModelOutput() {
  }

  // constructor básico
  public RedModelOutput(ArrayList<int []> predictedRightClass) { //, int[] predictions
    this.predictedRightClass = predictedRightClass;
  }
 
  public ArrayList<int []> getOut() {
    return predictedRightClass;
  }

  /*int[] getPredictions() {
    return predictions;
  }
*/
  @Override
  public void readFields(DataInput in) throws IOException {
   // boolean readRuleBase = in.readBoolean();
  // leer tamaño:
	  System.out.println("\n*****************\nEstoy leyendo\n*****************");

	int size = in.readInt();

	predictedRightClass = new ArrayList<int []>();
	
	for(int i = 0 ; i < size ; i++){
		int[] aux = new int[2];
    	aux[0] = in.readInt();
    	predictedRightClass.add(aux);
       	predictedRightClass.get(i)[1] = in.readInt();
	}
  }

  @Override
  public void write(DataOutput out) throws IOException {
	  System.out.println("\n*****************\nEstoy escribiendo\n*****************");
	int size = predictedRightClass.size();
	out.writeInt(size);

	for(int i = 0 ; i < size ; i++){
		out.writeInt(predictedRightClass.get(i)[0]);
		out.writeInt(predictedRightClass.get(i)[1]);
	}

  }

  @Override
  public RedModelOutput clone() {
    return new RedModelOutput(predictedRightClass); //, predictions
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof RedModelOutput)) {
      return false;
    }

    RedModelOutput mo = (RedModelOutput) obj;

    return ((predictedRightClass == null && mo.getOut() == null) || (predictedRightClass != null && predictedRightClass.equals(mo.getOut()))); //&& Arrays.equals(predictions, mo.getPredictions()
  }

  /*
  @Override
  public int hashCode() {
    int hashCode = RS == null ? 1 : RS.hashCode();
    for (int prediction : predictions) {
      hashCode = 31 * hashCode + prediction;
    }
    return hashCode;
  }
  

  @Override
  public String toString() {
    return "{" + RS + " | " + Arrays.toString(predictions) + '}';
  }
  */

}

