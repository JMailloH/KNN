package org.apache.mahout.classifier.KnnMR.mapreduce;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.KnnMR.data.Data;
import org.apache.mahout.classifier.KnnMR.data.DataConverter;
import org.apache.mahout.classifier.KnnMR.data.DataLoader;
import org.apache.mahout.classifier.KnnMR.data.Dataset;
import org.apache.mahout.classifier.KnnMR.data.Instance;
import org.apache.mahout.classifier.KnnMR.utils.KnnMRUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.StringUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Mapreduce implementation that classifies with KNN the Input data using a previously preprocessed data
 */

public class KNNClassifier {
	
  private static final Logger log = LoggerFactory.getLogger(KNNClassifier.class);
  private final Path preprocessedPath;
  private final Path inputPath;
  private final Path datasetPath;

  private final Configuration conf;
  private final String cabecera;

  private final Path outputPath; // path that will containt the final output of the classifier
  private final Path mappersOutputPath; // mappers will output here
  private double[][] results;
	  
  public double[][] getResults() {
    return results;
  }
  
  // orden: dataset, info, RS, salida
  
  public KNNClassifier(Path inputPath, Path datasetPath, Path preprocessedPath, Path outputPath, Configuration conf, String cabecera) {
    this.inputPath = inputPath;
    this.preprocessedPath = preprocessedPath;
    this.datasetPath = datasetPath;
    this.outputPath = outputPath;
    this.conf = conf;
    this.cabecera=cabecera;

    mappersOutputPath = new Path(outputPath, "mappers");
  }
  
  private void configureJob(Job job) throws IOException {
    job.setJarByClass(KNNClassifier.class);

	FileInputFormat.setInputPaths(job, inputPath);
	FileOutputFormat.setOutputPath(job, mappersOutputPath);

	job.setOutputKeyClass(DoubleWritable.class);
	job.setOutputValueClass(Text.class);

	job.setMapperClass(ClassifierMapper.class);
	job.setNumReduceTasks(0); // no reducers

	job.setInputFormatClass(ClassifierTextInputFormat.class);
	job.setOutputFormatClass(SequenceFileOutputFormat.class);
  }

  /**
   * Mandatory to send the header to the mappers.
   * 
   * @param conf
   * @param header
   */
  private static void setHeader(Configuration conf, String header) {
	    conf.set("mahout.fc.InstanceSet", StringUtils.toString(header));
  }
  
  public static String getHeader(Configuration conf){
	    String string = conf.get("mahout.fc.InstanceSet");
	    if (string == null) {
	      return null;
	    }
	    
	    return StringUtils.fromString(string); 
  }
  
  public void run() throws IOException, ClassNotFoundException, InterruptedException {
    FileSystem fs = FileSystem.get(conf);

	// check the output
	if (fs.exists(outputPath)) {
	  throw new IOException("KNN: Output path already exists : " + outputPath);
	}

	
    setHeader(conf, cabecera);

    
	log.info("KNN: Adding the dataset to the DistributedCache");
	// put the test set into the DistributedCache
	
	DistributedCache.addCacheFile(datasetPath.toUri(), conf);
	
	log.info("KNN: Adding the preprocessed dataset to the DistributedCache");
	DistributedCache.addCacheFile(preprocessedPath.toUri(), conf);

	Job job = new Job(conf, "KNN classifier: "+datasetPath.getName()+", "+this.inputPath.getName());

	log.info("KNN: Configuring the job...");
	configureJob(job);

	log.info("KNN: Running the job...");
	if (!job.waitForCompletion(true)) {
	  throw new IllegalStateException("KNN: Job failed!");
	}

	parseOutput(job);

	HadoopUtil.delete(conf, mappersOutputPath);
  }
  
  /**
   * Extract the prediction for each mapper and write them in the corresponding output file. 
   * The name of the output file is based on the name of the corresponding input file.
   * Will compute the ConfusionMatrix if necessary.
   */
  private void parseOutput(JobContext job) throws IOException {
    Configuration conf = job.getConfiguration();
    FileSystem fs = mappersOutputPath.getFileSystem(conf);

    Path[] outfiles = KnnMRUtils.listOutputFiles(fs, mappersOutputPath);

    // read all the output
    List<double[]> resList = new ArrayList<double[]>();
    for (Path path : outfiles) {
      FSDataOutputStream ofile = null;
      try {
        for (Pair<DoubleWritable,Text> record : new SequenceFileIterable<DoubleWritable,Text>(path, true, conf)) {
          double key = record.getFirst().get();
          String value = record.getSecond().toString();
          if (ofile == null) {
            // this is the first value, it contains the name of the input file
            ofile = fs.create(new Path(outputPath, value).suffix(".out"));
          } else {
            // The key contains the correct label of the data. The value contains a prediction
            ofile.writeChars(value); // write the prediction
            ofile.writeChar('\n');

            resList.add(new double[]{key, Double.valueOf(value)});
          }
        }
      } finally {
        Closeables.closeQuietly(ofile);
      }
    }
    results = new double[resList.size()][2];
    resList.toArray(results);
  }
  
  /**
   * TextInputFormat that does not split the input files. This ensures that each input file is processed by one single
   * mapper.
   */
  private static class ClassifierTextInputFormat extends TextInputFormat {
    @Override
    protected boolean isSplitable(JobContext jobContext, Path path) {
      return false;
    }
  }
  
  public static class ClassifierMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {

    /** used to convert input values to data instances */
    private DataConverter converter;
    private PrototypeSet RS;
   // private final Random rng = RandomUtils.getRandom();
    private boolean first = true;
    private final Text lvalue = new Text();
    private Dataset test, preprocessed;
    
    protected String header;

    
    private final DoubleWritable lkey = new DoubleWritable();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      super.setup(context);    //To change body of overridden methods use File | Settings | File Templates.

      Configuration conf = context.getConfiguration();
      
      header=KNNClassifier.getHeader(conf);
      
    //  log.info("cabecera: "+header);
      KnnMRUtils.readHeader(this.header);
      
      URI[] files = DistributedCache.getCacheFiles(conf);

      if (files == null || files.length < 2) {
        throw new IOException("not enough paths in the DistributedCache");
      }
       
      test = Dataset.load(conf, new Path(files[0].getPath()));

      context.progress();
      converter = new DataConverter(test);

      context.progress();

      RS = PrototypeSet.load(conf, new Path(files[1].getPath()));
      context.progress();

      // RS=PGUtils.readPrototypeSet(files[1].getPath().toString());
      
      // System.out.println("RS size = "+RS.size());
      log.info("RS size = "+RS.size());
     // RS.print();
      
      //System.out.println(files[1].getPath().toString());
      // Esto se usaría si guardo los RSs en formato Hadoop.
                 
     /* 
      preprocessed = Dataset.load(conf, new Path(files[1].getPath()));
      
      if (preprocessed == null) {
        throw new InterruptedException("preprocessed set not found!");
      }
      */
      
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      if (first) {
        FileSplit split = (FileSplit) context.getInputSplit();
        Path path = split.getPath(); // current split path
        lvalue.set(path.getName());
        lkey.set(key.get());
        context.write(lkey, lvalue);

        first = false;
      }

      String line = value.toString();
      if (!line.isEmpty()) {
        Instance instance = converter.convert(line);
        
        Prototype objetivo = new Prototype(instance.get()); // conversion to prototype
        
        //System.out.println("RS size"+RS.size());

        context.progress();
        Prototype aux= KNN._1nn(objetivo, RS, context);
        
        context.progress();

        
        double prediction = aux.getOutput(0);
       // aux.print();
        //System.out.println("prediction: "+prediction);
        
        lkey.set(test.getLabel(instance));
        lvalue.set(Double.toString(prediction));
        context.write(lkey, lvalue);
      }
    }
  }

}
