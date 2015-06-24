package org.apache.mahout.classifier.KnnMR.mapreduce;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.classifier.KnnMR.utils.Pair;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.locks.ReentrantReadWriteLock.ReadLock;


// Print a PrototypeSet! .

/**
 * Print a reduced set as a PrototypeSet.
 */
public class MapredOutput implements Writable, Cloneable {

  //private ArrayList<PrototypeSet> resultingSet; //Map output
  private ArrayList<int []> predictedRightClass; //Reduce output 
  private int map_red; //Say the type of output - 1 for map output | 2 for reduce output
  private long time; //Time of the run
  protected ArrayList<Long> timer; //As Reduce output, the run time mappers

  
  // Class-distance matrix. First component: Classes, Second: Distances
  private ArrayList<Pair <Double[], Double[]>> resultingSet = new ArrayList<Pair <Double[], Double[]>>();
  
  //	        Pair <Double[], Double[]> result= new Pair <Double[], Double[]>(distances, classes);
  
  
  public MapredOutput() {
  }

  public MapredOutput(int map_red) {
	  this.map_red = map_red;
  }
  
  /**
   * Basic constructor
   * 
   * @param input, output of the current action
   * @param map_red, 1 for map output - 2 for reduce output
   */
  public MapredOutput(ArrayList input, int map_red) { //, int[] predictions
	  this.map_red = map_red;
	  if (map_red == 1){
		  this.resultingSet = input;
	  }else{
		  this.predictedRightClass = input;
	  }
  //  this.predictions = predictions;
  }
  
  
  /**
   * Basic constructor
   * 
   * @param input, output of the current action
   * @param map_red, 1 for map output - 2 for reduce output
   * @param time, time of the run
   */
  public MapredOutput(ArrayList input, int map_red, long time) { //, int[] predictions
	  this.map_red = map_red;
	  this.time = time;
	  if (map_red == 1){
		  this.resultingSet = input;
	  }else{
		  this.predictedRightClass = input;
	  }
  //  this.predictions = predictions;
  }
  
  /**
   * Basic constructor
   * 
   * @param input, output of the current action
   * @param map_red, 1 for map output - 2 for reduce output
   * @param timer, list of the time's run map
   * @param timerReduce, list of reduce run time.
   * @param time, time of the run
   */
  public MapredOutput(ArrayList input, int map_red, ArrayList<Long> timer, long time) { 
	  this.map_red = map_red;
	  this.time = time;
	  this.timer = timer;
	  if (map_red == 1){
		  this.resultingSet = input;
	  }else{
		  this.predictedRightClass = input;
	  }
  //  this.predictions = predictions;
  }
 
  public ArrayList<Pair <Double[], Double[]>> getResultingSet() {
    return resultingSet;
  }

  public ArrayList<int []> getOut() {
	return predictedRightClass;
  }
  
  public long getTime() {
	return time;
  }
  
  public ArrayList<Long> getTimer() {
	return timer;
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
	this.map_red = in.readInt();

	if(map_red == 1){
		//Read the run time map
		this.time = in.readLong();
		
		//Read the partial solution
		int size = in.readInt();
		
		
		// read number of neighbors
		int neighbors = in.readInt();
		
		resultingSet=new ArrayList<Pair <Double[], Double[]>>();
		
		for(int i=0; i<size;i++){ // for all the test instances, read their class-distance pairs.
			
			Double classes[] = new Double[neighbors];
			Double distances[] = new Double[neighbors];
			
			for(int j=0; j<neighbors; j++){
				classes[j]=in.readDouble();
				distances[j]=in.readDouble();
			}
			
			Pair <Double[], Double[]> result = new Pair<Double[], Double[]>(classes,distances);
			resultingSet.add(result);
		}
		/*resultingSet = new ArrayList<PrototypeSet>();
		for(int i=0; i<size;i++){
			PrototypeSet aux = new PrototypeSet();
			aux.readFields(in);
			resultingSet.add(aux);
		}
		*/
	}else{
		//Read the reduce run time
		time = in.readLong();
		
		//Read run time of the map
		int size = in.readInt();
		timer = new ArrayList<Long>();
		for(int i = 0 ; i < size ; i++){
			timer.add(in.readLong());
		}

		//Read the final solution
		size = in.readInt();
		predictedRightClass = new ArrayList<int []>();
		for(int i = 0 ; i < size ; i++){
			int[] aux = new int[2];
	    	aux[0] = in.readInt();
	    	predictedRightClass.add(aux);
	       	predictedRightClass.get(i)[1] = in.readInt();
		}
	}

  }

  @Override
  public void write(DataOutput out) throws IOException {
	out.writeInt(map_red);
	
	if(map_red == 1){
		//Write the run time map
		out.writeLong(time);
		
		//Write the partial solution
		out.writeInt(resultingSet.size());
		
		// write number of neighbors:
		int neighbors= resultingSet.get(0).first().length;// number of neighbors
		out.writeInt(neighbors);
		
	//	System.out.println("WRITING HDFS: "+resultingSet.size()+","+neighbors);
		
		for( Pair <Double[], Double[]> result : resultingSet){ // for each pair Class-distance
			
			if(result!=null){
				
				for (int i=0; i<neighbors;i++){
					out.writeDouble(result.first()[i]);	 
					out.writeDouble(result.second()[i]);
				}
				
			}
		}
		/*
		for(PrototypeSet ps : resultingSet){
	        if (ps != null) {
				ps.write(out);
	
	        }
	    }
	    */
	}else{
		//Write the reduce run time
		out.writeLong(time);
		
		//Write run time of the map
		int size = timer.size();
		out.writeInt(size);
		for(int i = 0 ; i < size ; i++){
			out.writeLong(timer.get(i));
		}
		//Write the final solution
		size = predictedRightClass.size();
		out.writeInt(size);
		for(int i = 0 ; i < size ; i++){
			out.writeInt(predictedRightClass.get(i)[0]);
			out.writeInt(predictedRightClass.get(i)[1]);
		}
	}

  }

  @Override
  public MapredOutput clone() {
	MapredOutput output;

	if (map_red == 1){
		output = new MapredOutput(resultingSet,1); 
	}else{
		output = new MapredOutput(predictedRightClass,2); 
	}
	return output;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof MapredOutput)) {
      return false;
    }

    MapredOutput mo = (MapredOutput) obj;
    if(map_red == 1){
    	return ((resultingSet == null && mo.getResultingSet() == null) || (resultingSet != null && resultingSet.equals(mo.getResultingSet())));
    }else{
        return ((predictedRightClass == null && mo.getOut() == null) || (predictedRightClass != null && predictedRightClass.equals(mo.getOut())));
    }
  }


}

