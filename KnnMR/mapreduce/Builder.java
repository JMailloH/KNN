package org.apache.mahout.classifier.KnnMR.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.mahout.classifier.KnnMR.builder.IBLclassifier;
import org.apache.mahout.classifier.KnnMR.data.Dataset;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringUtils;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public abstract class Builder {
  
  private static final Logger log = LoggerFactory.getLogger(Builder.class);
  
  private final Path dataPath;
  private final Path datasetPath;
  private final Configuration conf;
  private final String cabecera;
  private final String outputDirName;
  private final IBLclassifier classifier;
  private final int kneighbors;
  private final String testName;
  private String dataName;
  private final String reduce;
  
  protected Builder(IBLclassifier classifier, Path dataPath, Path datasetPath, Configuration conf, String cabecera, int kneighbors, String outputDirName, String testName, String reduce) {
    this.classifier = classifier;
	this.dataPath = dataPath;
    this.datasetPath = datasetPath;
    this.cabecera=cabecera;
    this.outputDirName=outputDirName;
    this.kneighbors=kneighbors;
    this.conf = new Configuration(conf);
    this.testName = testName;
    this.reduce = reduce;
    this.dataName = dataPath.toString();
  }
    
  protected IBLclassifier getIBLclassifier(){
	    return classifier;  
  }
  
  protected String getTestName(){
	    return testName;  
  }
  
  protected Path getDataPath() {
    return dataPath;
  }
  
  protected String getDataName(){
	    return dataName;  
  }
  
  protected String getOutputName(){
	    return outputDirName;  
  }
  /**
   * Return the value of "mapred.map.tasks".
   * 
   * @param conf
   *          configuration
   * @return number of map tasks
   */
  public static int getNumMaps(Configuration conf) {
    return conf.getInt("mapred.map.tasks", -1);
  }

  /**
   * Used only for DEBUG purposes. if false, the mappers doesn't output anything, so the builder has nothing
   * to process
   * 
   * @param conf
   *          configuration
   * @return true if the builder has to return output. false otherwise
   */
  protected static boolean isOutput(Configuration conf) {
    return conf.getBoolean("debug.mahout.fc.output", true);
  }
  
  public static int getKNeighbour(Configuration conf) {
    String string = conf.get("mahout.fc.KNeighbour");
    if (string == null) {
      return (Integer) null;
    }
   
    return StringUtils.fromString(string);
  }
  
  public static IBLclassifier getIBLclassifierBuilder(Configuration conf) {
	    String string = conf.get("mahout.fc.classifier");
	    if (string == null) {
	      return null;
	    }
	    
	    return StringUtils.fromString(string);
	  }
  
  
  public static String getHeader(Configuration conf){
	    String string = conf.get("mahout.fc.InstanceSet");
	    if (string == null) {
	      return null;
	    }
	    
	    return StringUtils.fromString(string); 
  }
  
  public static String getOutput(Configuration conf){
	    String string = conf.get("mahout.fc.OutputName");
	    if (string == null) {
	      return null;
	    }
	    
	    return string; 
  }
  
  public static String getTestName(Configuration conf){
	    String string = conf.get("mahout.fc.TestName");
	    if (string == null) {
	      return null;
	    }
	    
	    return StringUtils.fromString(string); 
  }
 
  public static String getDataName(Configuration conf){
	    String string = conf.get("mahout.fc.DataName");
	    if (string == null) {
	      return null;
	    }
	    
	    return StringUtils.fromString(string); 
  }  
  
  public static String getReduce(Configuration conf){
	    String string = conf.get("mahout.fc.reduce");
	    if (string == null) {
	      return null;
	    }
	    
	    return StringUtils.fromString(string); 
  }
  
  private static void setIBLclassifierBuilder(Configuration conf, IBLclassifier classifier) {
	    conf.set("mahout.fc.classifier", StringUtils.toString(classifier));
  }
  
  private static void setKNeighbour(Configuration conf, int k) {
    conf.set("mahout.fc.KNeighbour", StringUtils.toString(k));
  }
  
  private static void setHeader(Configuration conf, String header) {
	    conf.set("mahout.fc.InstanceSet", StringUtils.toString(header));
  }
  
  private static void setTestName(Configuration conf, String testName) {
	    conf.set("mahout.fc.TestName", StringUtils.toString(testName));
  }
  
  private static void setDataName(Configuration conf, String dataName) {
	    conf.set("mahout.fc.DataName", StringUtils.toString(dataName));
  }
  
  private static void setReduce(Configuration conf, String reduce) {
	    conf.set("mahout.fc.reduce", StringUtils.toString(reduce));
  }
  
  private static void setOutputName(Configuration conf, String outputDirName) {
	    conf.set("mahout.fc.OutputName", outputDirName);
  }
  /**
   * Output Directory name
   * 
   * @param conf
   *          configuration
   * @return output dir. path (%WORKING_DIRECTORY%/OUTPUT_DIR_NAME%)
   * @throws IOException
   *           if we cannot get the default FileSystem
   */
  protected Path getOutputPath(Configuration conf) throws IOException {
    // the output directory is accessed only by this class, so use the default
    // file system
    FileSystem fs = FileSystem.get(conf);
    return new Path(fs.getWorkingDirectory(), outputDirName);
  }
  
  /**
   * Helper method. Get a path from the DistributedCache
   * 
   * @param conf
   *          configuration
   * @param index
   *          index of the path in the DistributedCache files
   * @return path from the DistributedCache
   * @throws IOException
   *           if no path is found
   */
  public static Path getDistributedCacheFile(Configuration conf, int index) throws IOException {
    URI[] files = DistributedCache.getCacheFiles(conf);
    
    if (files == null || files.length <= index) {
      throw new IOException("path not found in the DistributedCache");
    }
    
    return new Path(files[index].getPath());
  }
  
  /**
   * Helper method. Load a Dataset stored in the DistributedCache
   * 
   * @param conf
   *          configuration
   * @return loaded Dataset
   * @throws IOException
   *           if we cannot retrieve the Dataset path from the DistributedCache, or the Dataset could not be
   *           loaded
   */
  public static Dataset loadDataset(Configuration conf) throws IOException {
    Path datasetPath = getDistributedCacheFile(conf, 0);
    
    return Dataset.load(conf, datasetPath);
  }
  
  /**
   * Used by the inheriting classes to configure the job
   * 
   *
   * @param job
   *          Hadoop's Job
   * @throws IOException
   *           if anything goes wrong while configuring the job
   */
  protected abstract void configureJob(Job job) throws IOException;
  
  /**
   * Sequential implementation should override this method to simulate the job execution
   * 
   * @param job
   *          Hadoop's job
   * @return true is the job succeeded
   */
  protected boolean runJob(Job job) throws ClassNotFoundException, IOException, InterruptedException {
    return job.waitForCompletion(true);
  }
  
  /**
   * Parse the output files to extract the model and pass the predictions to the callback
   * 
   * @param job
   *          Hadoop's job
   * @return Built DecisionForest
   * @throws IOException
   *           if anything goes wrong while parsing the output
   */
  
   
  protected abstract MapredOutput parseOutput(Job job) throws IOException;
  
  
  public MapredOutput build() throws IOException, ClassNotFoundException, InterruptedException {
    
    Path outputPath = getOutputPath(conf);
    //FileSystem fs = outputPath.getFileSystem(conf);
    
    
    // check the output
    /*
    if (fs.exists(outputPath)) {
      throw new IOException("PG: Output path already exists : " + outputPath);
    }
    */
    setReduce(conf, reduce);
    setTestName(conf, testName);
    setDataName(conf, dataName);
    setIBLclassifierBuilder(conf, classifier);
    setKNeighbour(conf, kneighbors);
    setOutputName(conf, outputDirName);
    // to send the header instance set.
    setHeader(conf, cabecera);
    // put the dataset into the DistributedCache
    DistributedCache.addCacheFile(datasetPath.toUri(), conf);
    
     
    Job job = new Job(conf, "Classifier: "+classifier.classifier+" - Number of K neighbors: "+kneighbors+" "+" - DataPath: "+dataPath.getName());
    		//PG builder;"+pg_algorithm.PGmethod+" "+dataPath.getName()+", "+this.reducePhase);
        
    log.debug("KnnMR: Configuring the job...");
    configureJob(job);
    
    log.debug("KnnMR: Running the job...");
    if (!runJob(job)) {
      log.error("KnnMR: Job failed!");
      return null;
    }
    
   
    if (isOutput(conf)) {
      log.debug("KnnMR: Parsing the output...; converting to RS");
      MapredOutput resultingSet = parseOutput(job);
      HadoopUtil.delete(conf, outputPath);
      return resultingSet;
    }
    
    return null;
    
  }
  
  /**
   * sort the splits into order based on size, so that the biggest go first.<br>
   * This is the same code used by Hadoop's JobClient.
   * 
   * @param splits
   *          input splits
   */
  public static void sortSplits(InputSplit[] splits) {
    Arrays.sort(splits, new Comparator<InputSplit>() {
      @Override
      public int compare(InputSplit a, InputSplit b) {
        try {
          long left = a.getLength();
          long right = b.getLength();
          if (left == right) {
            return 0;
          } else if (left < right) {
            return 1;
          } else {
            return -1;
          }
        } catch (IOException ie) {
          throw new IllegalStateException("Problem getting input split size", ie);
        } catch (InterruptedException ie) {
          throw new IllegalStateException("Problem getting input split size", ie);
        }
      }
    });
  }
}

