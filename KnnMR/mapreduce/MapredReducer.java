package org.apache.mahout.classifier.KnnMR.mapreduce;

import com.google.common.base.Preconditions;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.KnnMR.data.Dataset;
import org.apache.mahout.classifier.KnnMR.mapreduce.partial.StrataID;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected String reduce;

  
  private Dataset dataset;
  protected String header;
  protected int sizeTestSet;

  protected PrototypeSet join = new PrototypeSet();
  protected int strata;

  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected int getSizeTestSet() {
	    return sizeTestSet;
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
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getReduce(conf), Builder.loadDataset(conf), Builder.getHeader(conf));
    }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, String reduce, Dataset dataset, String header) {
    this.noOutput = noOutput;
    this.reduce = reduce;
    this.dataset = dataset;
    this.header = header;
  }

  
  /**
   * Generic reducer, it only adds all the RSs.
   */
  
protected void reduce(KEYIN id, Iterable<VALUEIN> rs, Context context)
		throws IOException, InterruptedException {
	// TODO Apéndice de método generado automáticamente
	
	//System.out.println("Si paso por aquí: "+id);
	//strata = (StrataID) id;

	for(VALUEIN value: rs){
		context.progress();
		MapredOutput prueba = (MapredOutput) value;
		ArrayList<PrototypeSet> strato = prueba.getResultingSet();
	
		//join.add(strato);
	}
	
	// if you write here, the cleanup does not work.
	//MapredOutput salida= new MapredOutput(join);
	//context.write((KEYOUT) id, (VALUEOUT) salida);

}


}


