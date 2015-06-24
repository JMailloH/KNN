package org.apache.mahout.classifier.KnnMR.mapreduce;

import com.google.common.base.Preconditions;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
//import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.KnnMR.builder.IBLclassifier;
import org.apache.mahout.classifier.KnnMR.data.Dataset;
import org.apache.mahout.classifier.KnnMR.mapreduce.Builder;
//import org.apache.mahout.keel.Dataset.InstanceSet;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import java.io.IOException;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredMapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Mapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected IBLclassifier classifier;
  protected int Kneighbour;
  protected String testName;
  protected String dataName;
  protected String header;
  protected String reduce;
  protected String outputName;
  private Dataset dataset;
  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected IBLclassifier getIBLclassifierBuilder() {
	    return classifier;
  }
  
  protected int getKNeighbour() {
    return Kneighbour;
  }
  
  protected String getDataName() {
    return dataName;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  protected String getInstanceSet() {
	return header;
  }
  
  /*
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Integer.parseInt(Builder.getKNeighbour(conf)), Builder.loadDataset(conf), Builder.getHeader(conf));
  }
  */
/**
   * Useful for testing
   */
  /*
  protected void configure(boolean noOutput, int KNeighbour, Dataset dataset, String header) {
    this.noOutput = noOutput;
    this.Kneighbour = KNeighbour;
    this.dataset = dataset;
    this.header = header;
  }
*/
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    Kneighbour = Builder.getKNeighbour(conf);
    classifier = Builder.getIBLclassifierBuilder(conf);
    reduce = Builder.getReduce(conf);
    dataName = Builder.getDataName(conf);
    outputName = Builder.getOutput(conf);
    
    
    configure(!Builder.isOutput(conf), Builder.getTestName(conf), Builder.loadDataset(conf), Builder.getHeader(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, String testName, Dataset dataset, String header) {
    Preconditions.checkArgument(classifier != null, "Classifier not found in the Job parameters");
    this.noOutput = noOutput;
    this.testName = testName;
    this.dataset = dataset;
    this.header = header;
  }
  
  
  
  

}


