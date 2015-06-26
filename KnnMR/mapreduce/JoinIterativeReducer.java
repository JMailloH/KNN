package org.apache.mahout.classifier.KnnMR.mapreduce;

import com.google.common.base.Preconditions;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.KnnMR.builder.IBLclassifier;
import org.apache.mahout.classifier.KnnMR.data.DataConverter;
import org.apache.mahout.classifier.KnnMR.data.Dataset;
import org.apache.mahout.classifier.KnnMR.mapreduce.partial.StrataID;
import org.apache.mahout.classifier.KnnMR.utils.KnnMRUtils;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.classifier.KnnMR.utils.Pair;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;
import org.apache.mahout.keel.Dataset.Instance;
import org.apache.mahout.keel.Dataset.InstanceSet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.Vector;

/**
 * This Mapred allows to run more than one reducers.
 * 
 */
public class JoinIterativeReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
	  private boolean noOutput;
	  
	  protected String reduce;
	  protected String outputDir;
	  
	  private Dataset dataset;
	  protected String header;
	  protected String testName;
	  protected IBLclassifier classifier;
	  protected int Kneighbour;

	 // protected ArrayList<PrototypeSet> join;
	  protected ArrayList<Pair <Double[], Double[]>> classDistanceMatrix;
	  Map<Integer, ArrayList<Pair <Double, Double>>> classDistanceResult = new TreeMap<Integer, ArrayList<Pair <Double, Double>>>();
	  protected ArrayList<Pair <Double[], Double[]>> vecinos;
	  protected ArrayList<Long> timer;
	  protected long time;
	  protected int strata;
	  private int firstId = 0;

	  
	  /**
	   * 
	   * @return if false, the mapper does not estimate and output predictions
	   */
	  protected boolean isNoOutput() {
	    return noOutput;
	  }
	  
	  protected String getTestName() {
	    return testName;
	  }
	  
	  protected Dataset getDataset() {
	    return dataset;
	  }
	  
	  protected String getReduce(){
		  return reduce;
	  }
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    time = System.currentTimeMillis();

    Configuration conf = context.getConfiguration();
    configure(!Builder.isOutput(conf), Builder.getReduce(conf), Builder.loadDataset(conf), Builder.getHeader(conf), Builder.getTestName(conf),Builder.getIBLclassifierBuilder(conf),Builder.getKNeighbour(conf),Builder.getOutput(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, String reduce, Dataset dataset, String header, String testName, IBLclassifier classifier, int Kneighbour, String outputDir) {
    this.noOutput = noOutput;
    this.reduce = reduce;
    this.dataset = dataset;
    this.header = header;
    this.testName = testName;
    this.classifier = classifier;
    this.Kneighbour = Kneighbour;
    this.outputDir = outputDir;
    //this.join = new ArrayList<PrototypeSet>();
    this.classDistanceMatrix = new ArrayList<Pair <Double[], Double[]>>();
    this.vecinos = new ArrayList<Pair <Double[], Double[]>>();

    
    this.timer = new ArrayList<Long>();
  }
  
  /**
   * Generic reducer, it only adds all the RSs.
   */
  
	protected void reduce(KEYIN id, Iterable<VALUEIN> rs, Context context) throws IOException, InterruptedException {

		long tiempo = System.currentTimeMillis();

		double first;
		double second;
		
		if(this.reduce.equals("OPC1")){
		    if(this.classifier.classifier.equals("KNN")){
				for(VALUEIN value: rs){
					MapredOutput resultados = (MapredOutput) value;
					context.progress();
					
					timer.add(resultados.getTime());
					
					ArrayList<Pair <Double[], Double[]>> resultingSet =resultados.getResultingSet();
					
					int key = Integer.parseInt(id.toString());
					
					//Si aún no ha llegado ningúnos vecinos para esa instancia, lo inicializamos.
					if(classDistanceResult.get(key)==null){
						classDistanceResult.put((Integer) key, new ArrayList<Pair <Double, Double>>());

						for (int i = 0; i < resultingSet.size() ; i++){
					    	context.progress();
						
					    	Pair<Double[], Double[]> auxPair = resultingSet.get(i);
					    	//classDistanceResult
					    	
					    	for (int j = 0 ; j < this.Kneighbour ; j++){
						    	first = auxPair.first()[j];
								second = auxPair.second()[j];
								classDistanceResult.get(key).add(new Pair <Double, Double>(first, second));
					    	}   	
						}
						
					//Actualizamos los vecinos más cercanos en caso de ser necesario.
					}else{
						// update the matrix when necessary:
						int sizeTS = resultingSet.size();
						int neighbors = classDistanceResult.get(key).size();
						//int neighbors = classDistanceMatrix.get(0).first().length;

						
						for(int r = 0 ; r < sizeTS ; r++){
							int x = 0;
							for(int i = 0 ; i< neighbors ; i++){
								if(resultingSet.get(r).second()[x]<classDistanceResult.get(key).get(i).second()){
									classDistanceResult.get(key).get(i).set(resultingSet.get(r).first()[x], resultingSet.get(r).second()[x]);
									x++;
								}
							}
						}
					}//Fin else			
					
				}
		    }
		} 
		
		//System.out.println("reduce procedure: " + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "");
		
	}


	 protected void cleanup(Context context) throws IOException, InterruptedException {
		 
		 int keyID = Integer.parseInt(context.getCurrentKey().toString());
		 ArrayList<int []> idPredictedClass = new ArrayList<int []>();
		 
		 StrataID KEY = new StrataID();
		 KEY.set(strata, firstId + 1);
		 
		 Map<Integer, Integer> auxVote = new HashMap<Integer, Integer>();
		 
		 //Muestro el TreeMap
		 Iterator it = classDistanceResult.keySet().iterator();
		 while(it.hasNext()){
			 int key = (Integer) it.next();
			 Collections.sort(classDistanceResult.get(key));
			 for(int i = 0 ; i < this.Kneighbour ; i++){
				 int classAux = classDistanceResult.get(key).get(i).first().intValue();
				 if(auxVote.get(classAux)==null){
					 auxVote.put(classAux,0);
				 }
				 auxVote.put(classAux,auxVote.get(classAux)+1);
				 //System.out.println("Clave: " + key + " -> Valor: " + classDistanceResult.get(key).get(i).first() + " " + classDistanceResult.get(key).get(i).second());
			 } 
			 
			 
			 int predictedClass = 0;
			 int vote = 0;
			 Iterator it2 = auxVote.keySet().iterator();
			 while(it2.hasNext()){
				 int key2 = (Integer) it2.next();
				 if(vote < auxVote.get(key2)){
					 vote = auxVote.get(key2);
					 predictedClass = key2;
				 }
			 }
			 
			//System.out.println("Clase predicha: " + predictedClass);	 
			//System.out.println("***********************" + outputDir + "************************");

			int[] aux = new int[2];
			aux[0] = key;
	    	aux[1] = predictedClass;
	    	idPredictedClass.add(aux);
	    	
			//in.append(key+"\t"+predictedClass+"\n");		    
	    	auxVote.clear();

		 }
		//inString = in.toString();
		//ofile.writeBytes(inString);
		//ofile.close();	
		
		 /*for(int i = 0 ; i < idPredictedClass.size() ; i++){
			 System.out.println(idPredictedClass.get(i)[0] + " " + idPredictedClass.get(i)[1]);
		 }*/
		
	    this.time = System.currentTimeMillis() - this.time;
	    MapredOutput salida= new MapredOutput(idPredictedClass,2,timer,this.time);
		context.write((KEYOUT) KEY, ((VALUEOUT) salida));

	 }
	 
}


