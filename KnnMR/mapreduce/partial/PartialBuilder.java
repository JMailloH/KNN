package org.apache.mahout.classifier.KnnMR.mapreduce.partial;

//import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
//import org.apache.mahout.classifier.KnnMR.*;
import org.apache.mahout.classifier.KnnMR.builder.IBLclassifier;
import org.apache.mahout.classifier.KnnMR.mapreduce.*;
import org.apache.mahout.classifier.KnnMR.utils.KnnMRUtils;
import org.apache.mahout.classifier.KnnMR.mapreduce.partial.JoinReducer;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import java.io.IOException;
//import java.util.Arrays;
import java.util.ArrayList;

/**
 * Builds a model using partial data. Each mapper uses only the data given by its InputSplit
 */
public class PartialBuilder extends Builder {

  public PartialBuilder(IBLclassifier classifier, Path dataPath, Path datasetPath, String cabecera, int kneighbors, String outputPath, String testName, String reduce) {
    this(classifier, dataPath, datasetPath, new Configuration(), cabecera, kneighbors, outputPath, testName, reduce);
  }
  
  public PartialBuilder(IBLclassifier classifier, Path dataPath, Path datasetPath, Configuration conf, String cabecera, int kneighbors, String outputPath,  String testName, String reduce) {
    super(classifier, dataPath, datasetPath, conf, cabecera, kneighbors, outputPath, testName, reduce);
  }

  @Override
  protected void configureJob(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    job.setJarByClass(PartialBuilder.class);
    
    FileInputFormat.setInputPaths(job, getDataPath());
    FileOutputFormat.setOutputPath(job, getOutputPath(conf));
    
    job.setOutputKeyClass(StrataID.class);

    job.setOutputValueClass(MapredOutput.class);
    
    job.setMapperClass(KnnMapper.class);
     
 // Elegir el JoinReducer
    job.setReducerClass(JoinReducer.class); 

   // job.setNumReduceTasks(10);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
  }
  
  
  
  @Override
  protected MapredOutput parseOutput(Job job) throws IOException {

    Configuration conf = job.getConfiguration();
    
    // int numMaps = Builder.getNumMaps(conf);
    
    Path outputPath = getOutputPath(conf);
          
    return processOutput(job, outputPath);
  }
  
  
  /**
   * Processes the output from the output path.<br>
   * 
   * @param outputPath
   *          directory that contains the output of the job
   * @param keys
   *          can be null
   * @param trees
   *          can be null
   * @throws java.io.IOException
   */
  
  
  protected MapredOutput processOutput(JobContext job, Path outputPath) throws IOException { 
    Configuration conf = job.getConfiguration();

    FileSystem fs = outputPath.getFileSystem(conf);

    Path[] outfiles = KnnMRUtils.listOutputFiles(fs, outputPath);
    
    //read the output (un solo fichero por ser reduce)
    MapredOutput value=null;
    for (Path path : outfiles) {
      for (Pair<StrataID,MapredOutput> record : new SequenceFileIterable<StrataID, MapredOutput>(path, conf)) {
        value = record.getSecond();
      }
    }
    
    // cojo el último, porque es iterativo, ó el único que hay si 
    // lo hago todo con solo reduce.
    
	return value;

  }
  
  
}

